#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Review System - Виправлена версія (Quick Fix)
"""

import http.server
import socketserver
import json
import os
import time
import glob
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional

class EnhancedVideoReviewHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Шляхи (конфігуруються через ENV)
        # COMFY_OUTPUT_DIR=/workspace/ComfyUI/output/
        # AGENT_T2I2V_DIR=/workspace/Agent_T2I2V/
        self.video_dir = os.environ.get("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output/")
        self.workspace_dir = os.environ.get("AGENT_T2I2V_DIR", "/workspace/Agent_T2I2V/")
        self.auto_state_dir = os.path.join(self.workspace_dir, "auto_state")
        
        # JSON файли системи навчання
        self.manual_ratings_file = os.path.join(self.auto_state_dir, "manual_ratings.json")
        self.bandit_state_file = os.path.join(self.auto_state_dir, "bandit_state.json")
        self.knowledge_file = os.path.join(self.auto_state_dir, "knowledge.json")
        self.review_queue_file = os.path.join(self.auto_state_dir, "review_queue.json")
        self.reference_params_file = os.path.join(self.auto_state_dir, "reference_params.json")
        
        # Забезпечуємо існування директорій
        os.makedirs(self.auto_state_dir, exist_ok=True)
        self._ensure_json_files()
        
        super().__init__(*args, **kwargs)
    
    def _ensure_json_files(self):
        """Створення початкових JSON файлів, якщо вони не існують"""
        if not os.path.exists(self.manual_ratings_file):
            self._save_json(self.manual_ratings_file, {})
        
        if not os.path.exists(self.bandit_state_file):
            self._save_json(self.bandit_state_file, {"arms": [], "N": [], "S": [], "t": 0})
        
        if not os.path.exists(self.knowledge_file):
            self._save_json(self.knowledge_file, {"best_score": 0, "best_params": {}, "history": []})
        
        if not os.path.exists(self.review_queue_file):
            self._save_json(self.review_queue_file, {"pending": [], "in_review": [], "completed": []})
    
    def _load_json(self, filepath: str, default=None):
        """Безпечне завантаження JSON"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Помилка завантаження {filepath}: {e}")
        return default if default is not None else {}
    
    def _save_json(self, filepath: str, data):
        """Безпечне збереження JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Помилка збереження {filepath}: {e}")
            return False

    def _append_reference_params(self, params: dict):
        """Додати запис у reference_params.json (масив об'єктів {"params": {...}})."""
        try:
            data = []
            if os.path.exists(self.reference_params_file):
                with open(self.reference_params_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
                    elif isinstance(loaded, dict):
                        for key in ("combos", "params_list", "list"):
                            if isinstance(loaded.get(key), list):
                                data = loaded.get(key)
                                break
            data.append({"params": params})
            with open(self.reference_params_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"⚠️ Не вдалось додати до reference_params.json: {e}")
            return False

    def _append_ban_history(self, record: dict):
        """Додати запис у ban_history.json (масив подій)."""
        path = os.path.join(self.auto_state_dir, "ban_history.json")
        try:
            hist = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        hist = loaded
            hist.append(record)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(hist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Не вдалось додати у ban_history.json: {e}")
    
    def do_GET(self):
        # Парсимо URL та параметри
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path
        
        print(f"🌐 GET запит: {self.path}")
        print(f"📍 Шлях: {path}")
        
        if path == '/':
            self.serve_main_page()
        elif path == '/api/videos':
            self.serve_videos_api()
        elif path == '/api/stats':
            self.serve_stats_api()
        elif path == '/api/debug':
            self.serve_debug_api()
        elif path.startswith('/image_file'):
            self.serve_image_file()
        elif path.startswith('/video/'):
            self.serve_video()
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/rate':
            self.handle_rating()
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        # Preflight support for CORS
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_main_page(self):
        """Повний HTML інтерфейс з усіма функціями"""
        html = """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Enhanced Video Review System - RunPod</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff; min-height: 100vh;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .header { 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 30px; margin-bottom: 30px; 
            text-align: center; border: 1px solid rgba(255,255,255,0.2);
        }
        
        .header h1 { 
            font-size: 2.5rem; margin-bottom: 10px; 
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; margin-bottom: 30px;
        }
        
        .stat-card { 
            background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
            border-radius: 12px; padding: 20px; text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-value { font-size: 2rem; font-weight: bold; margin-bottom: 5px; color: #ffd700; }
        .stat-label { font-size: 0.9rem; opacity: 0.8; }
        
        .video-section { 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 30px; margin-bottom: 20px;
        }
        
        .video-header { 
            display: flex; justify-content: space-between; align-items: center; 
            margin-bottom: 20px;
        }
        
        .video-nav { display: flex; gap: 10px; }
        
        .nav-btn { 
            background: #4CAF50; color: white; border: none; 
            padding: 10px 20px; border-radius: 8px; cursor: pointer; 
            font-size: 1rem; transition: all 0.3s;
        }
        
        .load-more-btn {
            background: #2196F3; color: white; border: none; 
            padding: 10px 20px; border-radius: 8px; cursor: pointer; 
            font-size: 1rem; transition: all 0.3s;
        }
        
        .nav-btn:hover, .load-more-btn:hover { 
            background: #45a049; transform: translateY(-2px); 
        }
        .nav-btn:disabled { background: #666; cursor: not-allowed; transform: none; }
        
        .video-container { 
            display: grid; grid-template-columns: 1fr 1fr; 
            gap: 30px; margin-bottom: 30px;
        }
        
        .video-player { 
            background: #000; border-radius: 12px; overflow: hidden;
        }
        
        .video-player video { width: 100%; height: auto; max-height: 400px; }
        
        .video-info { 
            background: rgba(255,255,255,0.05); border-radius: 12px; 
            padding: 20px;
        }
        
        .info-section { margin-bottom: 20px; }
        
        .info-title { 
            font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; 
            color: #ffd700;
        }
        
        .info-content { 
            background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; 
            font-family: 'Courier New', monospace; white-space: pre-wrap;
            max-height: 150px; overflow-y: auto; font-size: 0.9rem;
        }
        
        .params-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 10px;
        }
        
        .param-item { 
            background: rgba(0,0,0,0.3); padding: 8px 12px; 
            border-radius: 6px; text-align: center;
        }
        
        .param-label { font-size: 0.8rem; opacity: 0.7; margin-bottom: 2px; }
        .param-value { font-weight: bold; color: #ffd700; }
        
        .auto-metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); 
            gap: 8px; margin-top: 10px;
        }
        
        .metric-item { 
            background: rgba(0,100,255,0.2); padding: 6px; border-radius: 6px; 
            text-align: center; border: 1px solid rgba(0,100,255,0.3);
        }
        
        .metric-label { font-size: 0.75rem; opacity: 0.8; margin-bottom: 2px; }
        .metric-value { font-weight: bold; color: #87ceeb; font-size: 0.9rem; }
        
        .rating-form { 
            background: rgba(255,255,255,0.05); border-radius: 12px; 
            padding: 25px; margin-top: 20px;
        }
        
        .rating-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px;
        }
        
        .rating-group { 
            background: rgba(255,255,255,0.05); padding: 15px; 
            border-radius: 8px;
        }
        
        .rating-label { 
            display: block; font-weight: bold; margin-bottom: 10px; 
            color: #ffd700;
        }
        
        .rating-input { width: 100%; margin-bottom: 10px; }
        
        .rating-scale { 
            display: flex; justify-content: space-between; 
            font-size: 0.8rem; opacity: 0.7;
        }
        
        .rating-value { 
            font-size: 1.2rem; font-weight: bold; color: #ffd700; 
            text-align: center;
        }
        
        .defects-section { 
            grid-column: 1 / -1; background: rgba(255,0,0,0.1); 
            border: 1px solid rgba(255,0,0,0.3); border-radius: 8px; 
            padding: 15px;
        }
        
        .defects-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin-top: 10px;
        }
        
        .checkbox-group { display: flex; align-items: center; gap: 8px; }
        
        .comment-section { grid-column: 1 / -1; margin-top: 15px; }
        
        .comment-input { 
            width: 100%; min-height: 80px; padding: 12px; 
            border: 1px solid rgba(255,255,255,0.3); border-radius: 8px; 
            background: rgba(255,255,255,0.1); color: #fff; 
            font-family: inherit; resize: vertical;
        }
        
        .comment-input::placeholder { color: rgba(255,255,255,0.6); }
        
        .action-buttons { 
            display: flex; gap: 15px; justify-content: center; 
            margin-top: 25px;
        }
        
        .submit-btn { 
            background: linear-gradient(45deg, #4CAF50, #45a049); 
            color: white; border: none; padding: 15px 30px; 
            border-radius: 8px; cursor: pointer; font-size: 1.1rem; 
            font-weight: bold; transition: all 0.3s;
        }
        
        .submit-btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(76,175,80,0.4);
        }
        
        .skip-btn { 
            background: linear-gradient(45deg, #ff9800, #f57c00); 
            color: white; border: none; padding: 15px 30px; 
            border-radius: 8px; cursor: pointer; font-size: 1.1rem; 
            font-weight: bold; transition: all 0.3s;
        }
        
        .skip-btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(255,152,0,0.4);
        }
        
        .reference-checkbox { 
            display: flex; align-items: center; gap: 10px; 
            background: rgba(255,215,0,0.1); border: 1px solid rgba(255,215,0,0.3); 
            border-radius: 8px; padding: 15px; margin-top: 15px;
        }
        
        .no-videos { 
            text-align: center; padding: 60px 20px; 
            background: rgba(255,255,255,0.1); border-radius: 15px;
        }
        
        .no-videos h2 { 
            font-size: 2rem; margin-bottom: 15px; color: #ffd700;
        }
        
        .loading { 
            text-align: center; padding: 50px; 
            background: rgba(255,255,255,0.1); border-radius: 15px;
        }
        
        .debug-info {
            background: rgba(255,255,0,0.1); border: 1px solid rgba(255,255,0,0.3);
            border-radius: 8px; padding: 10px; margin: 10px 0; font-size: 0.8rem;
        }
        
        .error-message { 
            background: rgba(244,67,54,0.9); color: white; 
            padding: 10px 20px; border-radius: 8px; margin: 10px 0; 
            text-align: center;
        }
        
        .success-message { 
            background: rgba(76,175,80,0.9); color: white; 
            padding: 10px 20px; border-radius: 8px; margin: 10px 0; 
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .video-container { grid-template-columns: 1fr; }
            .rating-grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Enhanced Video Review System</h1>
            <p>Розумна система оцінки згенерованих відео з навчанням агента</p>
            <div id="stats-display">Завантажую статистику...</div>
        </div>

        <!-- Статистика -->
        <div id="stats-grid" class="stats-grid"></div>

        <!-- Секція відео -->
        <div id="video-section" class="loading">
            <h3>⏳ Завантаження відео...</h3>
        </div>
    </div>

    <script src="/static/review_app.js"></script>
    <!-- Inline fallback is retained below (commented). Keep external JS as primary. -->
    <!--
        let currentVideoIndex = 0;
        let videos = [];
        let stats = {};
        let videosPerPage = 50; // Зменшено для стабільності
        let loadedVideos = 0;

        // Завантаження початкових даних
        async function initializeApp() {
            console.log('🚀 Ініціалізація програми...');
            await Promise.all([loadStats(), loadVideos()]);
        }

        async function loadStats() {
            try {
                console.log('📊 Завантаження статистики...');
                const response = await fetch('/api/stats');
                stats = await response.json();
                console.log('📊 Статистика завантажена:', stats);
                renderStats();
            } catch (error) {
                console.error('❌ Помилка завантаження статистики:', error);
            }
        }

        async function loadVideos(offset = 0) {
            try {
                console.log(`🎬 Завантаження відео: offset=${offset}, limit=${videosPerPage}`);
                const url = `/api/videos?offset=${offset}&limit=${videosPerPage}`;
                console.log('🌐 URL запиту:', url);
                
                const response = await fetch(url);
                console.log('📡 Відповідь сервера:', response.status, response.statusText);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const newVideos = await response.json();
                console.log(`📦 Отримано відео: ${newVideos.length}`);
                
                if (offset === 0) {
                    videos = newVideos;
                } else {
                    videos = videos.concat(newVideos);
                }
                
                loadedVideos = offset + newVideos.length;
                console.log(`📈 Всього завантажено: ${loadedVideos}`);
                
                if (videos.length > 0) {
                    renderVideoInterface();
                    if (offset === 0) {
                        displayVideo(0);
                    }
                } else if (offset === 0) {
                    renderNoVideos();
                }
            } catch (error) {
                console.error('❌ Помилка завантаження відео:', error);
                document.getElementById('video-section').innerHTML = 
                    `<div class="error-message">❌ Помилка завантаження відео: ${error.message}</div>`;
            }
        }

        function renderStats() {
            const statsGrid = document.getElementById('stats-grid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_generated || 0}</div>
                    <div class="stat-label">Всього згенеровано</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.total_rated || 0}</div>
                    <div class="stat-label">Оцінено вручну</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.pending_count || 0}</div>
                    <div class="stat-label">Очікують оцінки</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.bandit_iterations || 0}</div>
                    <div class="stat-label">Ітерацій навчання</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stats.avg_rating || 0).toFixed(1)}</div>
                    <div class="stat-label">Середня оцінка</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.learning_arms || 0}</div>
                    <div class="stat-label">Варіантів навчання</div>
                </div>
            `;
        }

        function renderVideoInterface() {
            const loadMoreButton = loadedVideos < stats.pending_count ? 
                `<button class="load-more-btn" onclick="loadMoreVideos()">Завантажити ще ${Math.min(videosPerPage, stats.pending_count - loadedVideos)} відео</button>` : '';
            
            document.getElementById('video-section').innerHTML = `
                <div class="video-section">
                    <div class="video-header">
                        <h2>📹 Оцінка відео (<span id="current-index">1</span> з ${videos.length}) [Завантажено: ${loadedVideos}/${stats.pending_count}]</h2>
                        <div class="video-nav">
                            <button class="nav-btn" onclick="previousVideo()" id="prev-btn" disabled>⬅️ Попереднє</button>
                            <button class="nav-btn" onclick="nextVideo()" id="next-btn">Наступне ➡️</button>
                            ${loadMoreButton}
                        </div>
                    </div>
                    <div id="video-content"></div>
                </div>
            `;
        }

        async function loadMoreVideos() {
            await loadVideos(loadedVideos);
            renderVideoInterface();
        }

        function renderNoVideos() {
            document.getElementById('video-section').innerHTML = `
                <div class="no-videos">
                    <h2>🎉 Відмінно!</h2>
                    <p>Всі доступні відео вже оцінені. Система продовжує навчання на основі ваших оцінок.</p>
                    <p style="margin-top: 15px; opacity: 0.8;">Нові відео з'являться автоматично після генерації агентом.</p>
                </div>
            `;
        }

        function formatParams(video) {
            const params = [];
            if (video.fps && video.fps !== 'N/A') params.push(`<div class="param-item"><div class="param-label">FPS</div><div class="param-value">${video.fps}</div></div>`);
            if (video.width && video.height && video.width !== 'N/A') params.push(`<div class="param-item"><div class="param-label">Розмір</div><div class="param-value">${video.width}x${video.height}</div></div>`);
            if (video.seconds && video.seconds !== 'N/A') params.push(`<div class="param-item"><div class="param-label">Тривалість</div><div class="param-value">${video.seconds}с</div></div>`);
            if (video.seed && video.seed !== 'N/A') params.push(`<div class="param-item"><div class="param-label">Seed</div><div class="param-value">${video.seed}</div></div>`);
            if (video.combo && video.combo !== 'N/A') params.push(`<div class="param-item"><div class="param-label">Комбо</div><div class="param-value">${video.combo}</div></div>`);
            if (video.iteration && video.iteration !== 'N/A') params.push(`<div class="param-item"><div class="param-label">Ітерація</div><div class="param-value">${video.iteration}</div></div>`);
            
            return params.length > 0 ? `<div class="params-grid">${params.join('')}</div>` : '<div class="debug-info">⚠️ Параметри не знайдені в knowledge.json</div>';
        }

        function formatAutoMetrics(metrics) {
            if (!metrics || Object.keys(metrics).length === 0) {
                return '<div class="debug-info">⚠️ Автоматичні метрики не знайдені</div>';
            }
            
            const metricItems = [];
            for (const [key, value] of Object.entries(metrics)) {
                if (typeof value === 'number') {
                    metricItems.push(`
                        <div class="metric-item">
                            <div class="metric-label">${key}</div>
                            <div class="metric-value">${value.toFixed(2)}</div>
                        </div>
                    `);
                }
            }
            return metricItems.length > 0 ? `<div class="auto-metrics">${metricItems.join('')}</div>` : '<div class="debug-info">⚠️ Немає числових метрик</div>';
        }

        function displayVideo(index) {
            if (index < 0 || index >= videos.length) return;
            
            currentVideoIndex = index;
            const video = videos[index];
            const videoUrl = `/video/${encodeURIComponent(video.name)}`;
            
            // Debug інформація
            console.log('🎬 Відображення відео:', video);
            
            document.getElementById('video-content').innerHTML = `
                <div class="video-container">
                    <div class="video-player">
                        <video controls autoplay muted>
                            <source src="${videoUrl}" type="video/mp4">
                            Ваш браузер не підтримує відтворення відео.
                        </video>
                    </div>
                    <div class="video-info">
                        <div class="info-section">
                            <div class="info-title">📝 Промпти</div>
                            <div class="info-content"><strong>Video:</strong> ${video.prompt && video.prompt !== 'Промпт недоступний' ? video.prompt : '⚠️ Промпт не знайдений'}</div>
                            ${video.video_negative ? `<div class="info-content"><strong>Video negative:</strong> ${video.video_negative}</div>` : ''}
                            ${video.photo_prompt ? `<div class="info-content"><strong>Photo:</strong> ${video.photo_prompt}</div>` : ''}
                            ${video.photo_negative ? `<div class="info-content"><strong>Photo negative:</strong> ${video.photo_negative}</div>` : ''}
                        </div>
                        ${video.photo_path ? `
                        <div class="info-section">
                            <div class="info-title">📷 Фото (джерело)</div>
                            <div class="info-content">
                                <img src="/image_file?path=${encodeURIComponent(video.photo_path)}" style="max-width:100%; border-radius:8px;" />
                            </div>
                        </div>
                        ` : ''}
                        
                        <div class="info-section">
                            <div class="info-title">⚙️ Параметри генерації</div>
                            ${formatParams(video)}
                        </div>
                        
                        <div class="info-section">
                            <div class="info-title">🤖 Автоматичні метрики</div>
                            ${formatAutoMetrics(video.auto_metrics)}
                        </div>
                        
                        <div class="debug-info">
                            <strong>Debug:</strong> ${video.name} | Знайдено: ${video.found_in_knowledge ? 'Так' : 'Ні'} | Спосіб: ${video.match_method || 'Невідомо'}
                        </div>
                    </div>
                </div>

                <form class="rating-form" onsubmit="submitRating(event)">
                    <input type="hidden" name="video_name" value="${video.name}">
                    
                    <div class="rating-grid">
                        <div class="rating-group">
                            <label class="rating-label">🌟 Загальна якість</label>
                            <input type="range" name="overall_quality" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Дуже погано</span>
                                <span class="rating-value" id="overall_quality_value">5</span>
                                <span>10 - Відмінно</span>
                            </div>
                        </div>

                        <div class="rating-group">
                            <label class="rating-label">👁️ Візуальна якість</label>
                            <input type="range" name="visual_quality" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Розмито</span>
                                <span class="rating-value" id="visual_quality_value">5</span>
                                <span>10 - Кристально</span>
                            </div>
                        </div>

                        <div class="rating-group">
                            <label class="rating-label">🏃 Якість руху</label>
                            <input type="range" name="motion_quality" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Дьоргання</span>
                                <span class="rating-value" id="motion_quality_value">5</span>
                                <span>10 - Плавно</span>
                            </div>
                        </div>

                        <div class="rating-group">
                            <label class="rating-label">🎯 Відповідність промпту</label>
                            <input type="range" name="prompt_adherence" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Не відповідає</span>
                                <span class="rating-value" id="prompt_adherence_value">5</span>
                                <span>10 - Точно</span>
                            </div>
                        </div>

                        <div class="rating-group">
                            <label class="rating-label">🎨 Креативність</label>
                            <input type="range" name="creativity" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Банально</span>
                                <span class="rating-value" id="creativity_value">5</span>
                                <span>10 - Унікально</span>
                            </div>
                        </div>

                        <div class="rating-group">
                            <label class="rating-label">🔧 Технічна якість</label>
                            <input type="range" name="technical_quality" min="1" max="10" value="5" class="rating-input" oninput="updateRatingValue(this)">
                            <div class="rating-scale">
                                <span>1 - Артефакти</span>
                                <span class="rating-value" id="technical_quality_value">5</span>
                                <span>10 - Бездоганно</span>
                            </div>
                        </div>

                        <div class="defects-section">
                            <div class="info-title">⚠️ Виявлені дефекти</div>
                            <div class="defects-grid">
                                <div class="checkbox-group">
                                    <input type="checkbox" name="anatomy_issues" id="anatomy_issues">
                                    <label for="anatomy_issues">Проблеми з анатомією</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" name="extra_limbs" id="extra_limbs">
                                    <label for="extra_limbs">Зайві кінцівки</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" name="face_distortion" id="face_distortion">
                                    <label for="face_distortion">Спотворення обличчя</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" name="temporal_inconsistency" id="temporal_inconsistency">
                                    <label for="temporal_inconsistency">Непослідовність у часі</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" name="artifacts" id="artifacts">
                                    <label for="artifacts">Цифрові артефакти</label>
                                </div>
                                <div class="checkbox-group">
                                    <input type="checkbox" name="lighting_issues" id="lighting_issues">
                                    <label for="lighting_issues">Проблеми з освітленням</label>
                                </div>
                            </div>
                        </div>

                        <div class="comment-section">
                            <label class="rating-label" for="comments">💭 Коментарі та примітки</label>
                            <textarea name="comments" id="comments" class="comment-input" 
                                placeholder="Додаткові коментарі, спостереження або рекомендації для покращення..."></textarea>
                        </div>
                    </div>

                    <div class="reference-checkbox">
                        <input type="checkbox" name="is_reference" id="is_reference">
                        <label for="is_reference">⭐ Позначити як еталонне відео для навчання агента</label>
                    </div>

                    <div class="action-buttons">
                        <button type="submit" class="submit-btn">💾 Зберегти оцінку</button>
                        <button type="button" class="skip-btn" onclick="skipVideo()">⏭️ Пропустити</button>
                    </div>
                </form>
            `;

            // Оновлення навігації
            document.getElementById('current-index').textContent = index + 1;
            document.getElementById('prev-btn').disabled = index === 0;
            document.getElementById('next-btn').disabled = index === videos.length - 1;
        }

        function updateRatingValue(input) {
            const valueSpan = document.getElementById(input.name + '_value');
            if (valueSpan) {
                valueSpan.textContent = input.value;
            }
        }

        function previousVideo() {
            if (currentVideoIndex > 0) {
                currentVideoIndex--;
                displayVideo(currentVideoIndex);
            }
        }

        function nextVideo() {
            if (currentVideoIndex < videos.length - 1) {
                currentVideoIndex++;
                displayVideo(currentVideoIndex);
            }
        }

        function skipVideo() {
            nextVideo();
        }

        async function submitRating(event) {
            event.preventDefault();
            
            const formData = new FormData(event.target);
            const rating = {};
            
            // Збір всіх оцінок
            for (let [key, value] of formData.entries()) {
                if (key.includes('_issues') || key.includes('_limbs') || key.includes('_distortion') || 
                    key.includes('_inconsistency') || key.includes('_artifacts') || key.includes('_reference') ||
                    key.includes('lighting_issues')) {
                    rating[key] = true;
                } else if (key === 'comments') {
                    rating[key] = value;
                } else if (key !== 'video_name') {
                    rating[key] = parseFloat(value);
                }
            }

            // Додавання метаданих
            rating.rated_at = new Date().toISOString();
            rating.rated_by = 'manual_review';

            try {
                const response = await fetch('/api/rate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        video_name: formData.get('video_name'),
                        rating: rating
                    })
                });
                
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Показати повідомлення про успіх
                    const successDiv = document.createElement('div');
                    successDiv.className = 'success-message';
                    successDiv.textContent = '✅ Оцінка успішно збережена!';
                    document.getElementById('video-content').prepend(successDiv);
                    
                    // Видалення оціненого відео зі списку
                    videos.splice(currentVideoIndex, 1);
                    
                    // Оновлення статистики
                    await loadStats();
                    
                    setTimeout(() => {
                        if (videos.length === 0) {
                            renderNoVideos();
                        } else {
                            // Показ наступного відео або попереднього, якщо це було останнє
                            if (currentVideoIndex >= videos.length) {
                                currentVideoIndex = videos.length - 1;
                            }
                            displayVideo(currentVideoIndex);
                        }
                    }, 1000);
                } else {
                    throw new Error(result.message || 'Невідома помилка');
                }
            } catch (error) {
                console.error('❌ Помилка:', error);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = '❌ Помилка збереження: ' + error.message;
                document.getElementById('video-content').prepend(errorDiv);
            }
        }

        // Гарячі клавіші
        document.addEventListener('keydown', function(event) {
            if (event.ctrlKey || event.metaKey) return;
            
            switch(event.key) {
                case 'ArrowLeft':
                    event.preventDefault();
                    previousVideo();
                    break;
                case 'ArrowRight':
                    event.preventDefault();
                    nextVideo();
                    break;
                case 'Enter':
                    if (event.ctrlKey) {
                        event.preventDefault();
                        const form = document.querySelector('.rating-form');
                        if (form) {
                            form.dispatchEvent(new Event('submit'));
                        }
                    }
                    break;
            }
        });

        // Ініціалізація програми
        initializeApp();
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_videos_api(self):
        """API для отримання неоцінених відео з покращеним пошуком"""
        # Парсимо параметри запиту
        url_parts = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(url_parts.query)
        offset = int(query_params.get('offset', ['0'])[0])
        limit = int(query_params.get('limit', ['50'])[0])
        
        print(f"🔍 API відео: offset={offset}, limit={limit}")
        
        # Завантажуємо оцінені відео
        manual_ratings = self._load_json(self.manual_ratings_file, {})
        rated_videos = set(manual_ratings.keys())
        print(f"📊 Оцінені відео: {len(rated_videos)}")
        
        # Отримуємо всі відео файли
        video_files = glob.glob(f"{self.video_dir}*.mp4")
        video_files.sort(key=os.path.getmtime, reverse=True)
        print(f"🎬 Всього відео файлів: {len(video_files)}")
        
        # Завантажуємо knowledge для отримання деталей
        knowledge = self._load_json(self.knowledge_file, {"history": []})
        print(f"📚 Записів в knowledge.json: {len(knowledge.get('history', []))}")
        
        # Фільтруємо неоцінені відео
        unrated_videos = []
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            if video_name not in rated_videos:
                unrated_videos.append(video_path)
        
        print(f"⏳ Неоцінених відео: {len(unrated_videos)}")
        
        # Застосовуємо пагінацію
        paginated_videos = unrated_videos[offset:offset + limit]
        print(f"📦 Завантажуємо відео: {len(paginated_videos)}")
        
        result = []
        for video_path in paginated_videos:
            video_name = os.path.basename(video_path)
            stat = os.stat(video_path)
            
            # Покращений пошук деталей відео
            video_details, match_info = self._enhanced_video_search(video_name, knowledge)
            
            # Збираємо авто-метрики з knowledge (сумісно з новою структурою)
            auto_metrics = {}
            if isinstance(video_details, dict):
                try:
                    md = video_details.get('metrics') or {}
                    if isinstance(md, dict):
                        auto_metrics.update(md)
                except Exception:
                    pass
                try:
                    am = video_details.get('auto_metrics') or {}
                    if isinstance(am, dict):
                        auto_metrics.update(am)
                except Exception:
                    pass
                # Legacy плоскі поля як fallback
                if not auto_metrics:
                    for k in ('overall','blur','exposure','blockiness','flicker'):
                        v = video_details.get(k)
                        if v is not None:
                            auto_metrics[k] = v

            # Додаткові поля для фото/промптів
            photo_path = None
            photo_prompt = None
            photo_negative = None
            video_prompt = None
            video_negative = None
            if isinstance(video_details, dict):
                photo_path = video_details.get('source_image')
                photo_prompt = video_details.get('photo_prompt')
                photo_negative = video_details.get('photo_negative')
                video_prompt = video_details.get('prompt')
                video_negative = video_details.get('negative_prompt')

            video_info = {
                'name': video_name,
                'size_mb': round(stat.st_size / 1024 / 1024, 1),
                'created': time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                'prompt': video_prompt or video_details.get('prompt', 'Промпт недоступний'),
                'fps': video_details.get('params', {}).get('fps', 'N/A'),
                'width': video_details.get('params', {}).get('width', 'N/A'),
                'height': video_details.get('params', {}).get('height', 'N/A'),
                'seconds': video_details.get('params', {}).get('seconds', 'N/A'),
                'seed': video_details.get('params', {}).get('seed', 'N/A'),
                'combo': video_details.get('combo', 'N/A'),
                'auto_metrics': auto_metrics or {},                
                'iteration': video_details.get('iteration', 'N/A'),
                # Фото і промпти
                'photo_path': photo_path,
                'photo_prompt': photo_prompt,
                'photo_negative': photo_negative,
                'video_negative': video_negative,
                # Debug інформація
                'found_in_knowledge': match_info['found'],
                'match_method': match_info['method'],
                'search_details': match_info['details']
            }
            
            result.append(video_info)
        
        print(f"✅ Відправляємо {len(result)} відео")
        self.send_json_response(result)
    
    def serve_debug_api(self):
        """Debug API для перевірки стану системи"""
        knowledge = self._load_json(self.knowledge_file, {"history": []})
        manual_ratings = self._load_json(self.manual_ratings_file, {})
        
        # Приклади імен файлів для тестування
        video_files = glob.glob(f"{self.video_dir}*.mp4")[:10]
        
        debug_info = {
            "knowledge_entries": len(knowledge.get('history', [])),
            "manual_ratings": len(manual_ratings),
            "sample_video_files": [os.path.basename(f) for f in video_files],
            "sample_knowledge_entries": knowledge.get('history', [])[:3] if knowledge.get('history') else [],
            "sample_file_search": []
        }
        
        # Тестуємо пошук для кількох файлів
        for video_file in video_files[:3]:
            video_name = os.path.basename(video_file)
            details, match_info = self._enhanced_video_search(video_name, knowledge)
            debug_info["sample_file_search"].append({
                "video_name": video_name,
                "found": match_info['found'],
                "method": match_info['method'],
                "details": match_info['details'],
                "prompt_found": bool(details.get('prompt'))
            })
        
        self.send_json_response(debug_info)
    
    def _enhanced_video_search(self, video_name: str, knowledge: dict) -> tuple:
        """Покращений пошук відео в knowledge.json з debug інформацією"""
        match_info = {
            'found': False,
            'method': 'none',
            'details': {}
        }
        
        try:
            history = knowledge.get("history", [])
            if not history:
                match_info['details'] = {'error': 'Порожня історія в knowledge.json'}
                return {}, match_info
            
            print(f"🔍 Пошук відео: {video_name} в {len(history)} записах")
            
            # Метод 1: Пошук за точним ім'ям файлу
            for entry in history:
                # video_path = entry.get("video_path", "")
                video_path = entry.get("video") or entry.get("video_path") or ""
                if video_path and video_path.endswith(video_name):
                    print(f"✅ Знайдено за точним ім'ям: {video_path}")
                    match_info.update({
                        'found': True,
                        'method': 'exact_filename',
                        'details': {'matched_path': video_path}
                    })
                    return entry, match_info
            
            # Метод 2: Пошук за timestamp з імені файлу
            parts = video_name.split('_')
            if len(parts) >= 2:
                try:
                    timestamp_str = parts[1]
                    video_timestamp = int(timestamp_str)
                    print(f"🕐 Пошук за timestamp: {video_timestamp}")
                    
                    # Шукаємо найближчий за часом entry
                    best_match = None
                    min_diff = float('inf')
                    
                    for entry in history:
                        entry_timestamp = entry.get("timestamp", 0)
                        if entry_timestamp > 0:
                            diff = abs(entry_timestamp - video_timestamp)
                            if diff < min_diff:
                                min_diff = diff
                                best_match = entry
                    
                    if best_match and min_diff < 60:
                        print(f"✅ Знайдено за timestamp: diff={min_diff}s")
                        match_info.update({
                            'found': True,
                            'method': 'timestamp_match',
                            'details': {
                                'video_timestamp': video_timestamp,
                                'matched_timestamp': best_match.get("timestamp"),
                                'time_diff': min_diff
                            }
                        })
                        return best_match, match_info
                    
                except ValueError as e:
                    print(f"⚠️ Помилка парсингу timestamp: {e}")
            
            print(f"❌ Відео не знайдено: {video_name}")
            match_info['details'] = {
                'error': 'Не знайдено співпадінь',
                'video_name': video_name,
                'history_count': len(history)
            }
            
        except Exception as e:
            print(f"❌ Помилка пошуку: {e}")
            match_info['details'] = {'exception': str(e)}
        
        return {}, match_info
    
    def serve_stats_api(self):
        """API для отримання повної статистики системи"""
        try:
            # Завантажуємо всі JSON файли
            manual_ratings = self._load_json(self.manual_ratings_file, {})
            knowledge = self._load_json(self.knowledge_file, {"history": [], "best_score": 0})
            bandit_state = self._load_json(self.bandit_state_file, {"t": 0, "arms": []})
            
            # Підрахунок статистики
            total_generated = len(knowledge.get("history", []))
            total_rated = len(manual_ratings)
            
            # Підрахунок неоцінених відео
            video_files = glob.glob(f"{self.video_dir}*.mp4")
            rated_videos = set(manual_ratings.keys())
            pending_count = len([f for f in video_files if os.path.basename(f) not in rated_videos])
            
            # Середня оцінка
            avg_rating = 0
            if manual_ratings:
                overall_scores = []
                for rating_data in manual_ratings.values():
                    if isinstance(rating_data, dict) and 'rating' in rating_data:
                        rating = rating_data['rating']
                        if isinstance(rating, dict) and 'overall_quality' in rating:
                            overall_scores.append(rating['overall_quality'])
                
                if overall_scores:
                    avg_rating = sum(overall_scores) / len(overall_scores)
            
            stats = {
                "total_generated": total_generated,
                "total_rated": total_rated,
                "pending_count": pending_count,
                "avg_rating": avg_rating,
                "best_score": knowledge.get("best_score", 0),
                "bandit_iterations": bandit_state.get("t", 0),
                "learning_arms": len(bandit_state.get("arms", []))
            }
            
            print(f"📊 Статистика: Генеровано={total_generated}, Оцінено={total_rated}, Очікують={pending_count}")
            
        except Exception as e:
            print(f"⚠️ Помилка розрахунку статистики: {e}")
            stats = {
                "total_generated": 0,
                "total_rated": 0,
                "pending_count": 0,
                "avg_rating": 0,
                "best_score": 0,
                "bandit_iterations": 0,
                "learning_arms": 0
            }
        
        self.send_json_response(stats)
    
    def serve_video(self):
        """Відправка відео файлів"""
        video_name = urllib.parse.unquote(self.path.split('/')[-1])
        video_path = os.path.join(self.video_dir, video_name)
        
        if os.path.exists(video_path):
            self.send_response(200)
            self.send_header('Content-type', 'video/mp4')
            self.send_header('Content-Length', str(os.path.getsize(video_path)))
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            
            with open(video_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, f"Відео не знайдено: {video_name}")

    def serve_image_file(self):
        """Відправка зображення за абсолютним шляхом (довірене середовище)."""
        try:
            url_parts = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url_parts.query)
            p = (q.get('path', [''])[0] or '').strip()
            if not p or not os.path.exists(p):
                return self.send_json_response({'status': 'error', 'message': 'image path not found'})
            ct = 'image/jpeg'
            if p.lower().endswith('.png'):
                ct = 'image/png'
            self.send_response(200)
            self.send_header('Content-type', ct)
            self.send_header('Content-Length', str(os.path.getsize(p)))
            self.end_headers()
            with open(p, 'rb') as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_json_response({'status': 'error', 'message': str(e)})
    
    def handle_rating(self):
        """Обробка збереження оцінки з повною інтеграцією навчання"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            video_name = data.get('video_name')
            rating = data.get('rating')
            
            if not video_name or not rating:
                self.send_json_response({"status": "error", "message": "Неповні дані"})
                return
            
            # Завантажуємо поточні оцінки
            manual_ratings = self._load_json(self.manual_ratings_file, {})
            
            # Додаємо нову оцінку
            manual_ratings[video_name] = {
                "rating": rating,
                "timestamp": time.time(),
                "rated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Зберігаємо оновлені оцінки
            if self._save_json(self.manual_ratings_file, manual_ratings):
                print(f"✅ Збережена оцінка для відео: {video_name}")
                
                # Оновлюємо систему навчання
                self._update_learning_system(video_name, rating)

                # Якщо позначено як еталон — додамо у reference_params.json параметри з knowledge
                try:
                    if bool(rating.get('is_reference')):
                        knowledge = self._load_json(self.knowledge_file, {"history": []})
                        details, _mi = self._enhanced_video_search(video_name, knowledge)
                        params = (details.get('params') or {}) if isinstance(details, dict) else {}
                        # Fallback: пробуємо витягнути combo у вигляді полів
                        if not params and isinstance(details, dict):
                            combo = details.get('combo') or []
                            if isinstance(combo, list) and len(combo) >= 2:
                                params = {
                                    'sampler': combo[0],
                                    'scheduler': combo[1],
                                }
                        if isinstance(params, dict) and params:
                            # Додаємо seconds як безпечний дефолт
                            params.setdefault('seconds', 5.0)
                            self._append_reference_params(params)
                except Exception as e:
                    print(f"⚠️ Reference append failed: {e}")
                
                self.send_json_response({"status": "success"})
            else:
                self.send_json_response({"status": "error", "message": "Помилка збереження"})
                
        except Exception as e:
            print(f"❌ Помилка обробки оцінки: {e}")
            self.send_json_response({"status": "error", "message": str(e)})
    
    def _update_learning_system(self, video_name: str, rating: dict):
        """Оновлення системи навчання на основі ручної оцінки"""
        try:
            # Завантажуємо поточний стан
            bandit_state = self._load_json(self.bandit_state_file, {"arms": [], "N": [], "S": [], "t": 0})
            knowledge = self._load_json(self.knowledge_file, {"best_score": 0, "best_params": {}, "history": []})
            
            # Обчислюємо загальну оцінку
            overall_score = rating.get('overall_quality', 0)
            
            # Оновлюємо найкращий результат
            if overall_score > knowledge.get('best_score', 0):
                knowledge['best_score'] = overall_score
                
                # Знаходимо параметри цього відео
                video_details, _ = self._enhanced_video_search(video_name, knowledge)
                if video_details:
                    knowledge['best_params'] = {
                        'prompt': video_details.get('prompt', ''),
                        'params': video_details.get('params', {}),
                        'combo': video_details.get('combo', {}),
                        'manual_rating': rating
                    }
            
            # Оновлюємо bandit state
            bandit_state['t'] = bandit_state.get('t', 0) + 1
            
            # Додаємо інформацію про ручну оцінку до history
            for entry in knowledge.get("history", []):
                if entry.get("video_path", "").endswith(video_name):
                    entry['manual_rating'] = rating
                    entry['manual_rated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    break
            
            # Зберігаємо оновлені дані
            self._save_json(self.bandit_state_file, bandit_state)
            self._save_json(self.knowledge_file, knowledge)
            
            print(f"🎯 Оновлена система навчання для відео {video_name} з оцінкою {overall_score}")
            
        except Exception as e:
            print(f"⚠️ Помилка оновлення системи навчання: {e}")
    
    def send_json_response(self, data):
        """Відправка JSON відповіді"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))

class QAReviewHandler(EnhancedVideoReviewHandler):
    def do_POST(self):
        if self.path == '/api/ban_combo':
            self.handle_ban_combo()
            return
        return super().do_POST()
    
    def do_OPTIONS(self):
        # Preflight support for CORS
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        if self.path == '/qa':
            return self.serve_qa_page()
        if self.path.startswith('/search') and self.path == '/search':
            return self.serve_search_page()
        if self.path.startswith('/watch'):
            return self.serve_watch_page()
        if self.path == '/':
            # Render main page with integrated ban control
            return self.serve_main_page_with_ban()
        if self.path == '/base':
            # Serve original main UI from base handler
            return self.serve_main_page()
        if self.path.startswith('/api/videos'):
            # Augmented videos API with review_queue metrics
            return self.serve_videos_api_qa()
        if self.path.startswith('/api/search'):
            return self.serve_search_api()
        if self.path.startswith('/api/video_details'):
            return self.serve_video_details_api()
        return super().do_GET()

    def handle_ban_combo(self):
        try:
            content_length = int(self.headers.get('Content-Length', '0'))
            post_data = self.rfile.read(content_length) if content_length > 0 else b'{}'
            data = json.loads(post_data.decode('utf-8')) if post_data else {}

            combo_key = data.get('combo_key')
            params = data.get('params')
            video_name = data.get('video_name')

            bandit_path = self.bandit_state_file
            state = self._load_json(bandit_path, {"combo_stats": {}, "t": 0, "banned_combos": []})

            banned = set(state.get('banned_combos', []))

            # Resolve from knowledge using video_name if provided
            if not combo_key and video_name:
                knowledge = self._load_json(self.knowledge_file, {"history": []})
                history = knowledge.get("history", [])
                # try exact match by video or video_path
                for entry in history:
                    v = entry.get("video") or entry.get("video_path") or ""
                    if v.endswith(video_name):
                        combo = entry.get("combo") or []
                        entry_params = entry.get("params", {})
                        sampler = (combo[0] if isinstance(combo, list) and len(combo) > 0 else entry_params.get("sampler", "unknown"))
                        scheduler = (combo[1] if isinstance(combo, list) and len(combo) > 1 else entry_params.get("scheduler", "unknown"))
                        fps = str(entry_params.get("fps", 20))
                        cfg = str(entry_params.get("cfg_scale", entry_params.get("cfg", 7.0)))
                        steps = str(entry_params.get("steps", 25))
                        width = entry_params.get("width", 768)
                        height = entry_params.get("height", 432)
                        combo_key = f"{sampler}|{scheduler}|{fps}|{cfg}|{steps}|{width}x{height}"
                        break

            if not combo_key and params:
                # Rebuild combo_key like agent does: sampler|scheduler|fps|cfg|steps|WIDTHxHEIGHT
                sampler = params.get('sampler', 'unknown')
                scheduler = params.get('scheduler', 'unknown')
                fps = str(params.get('fps', 20))
                cfg = str(params.get('cfg_scale', 7.0))
                steps = str(params.get('steps', 25))
                width = params.get('width', 768)
                height = params.get('height', 432)
                combo_key = f"{sampler}|{scheduler}|{fps}|{cfg}|{steps}|{width}x{height}"

            if not combo_key:
                self.send_json_response({"status": "error", "message": "combo_key or params required"})
                return

            banned.add(combo_key)
            state['banned_combos'] = list(banned)

            self._save_json(bandit_path, state)

            # Also mark the video as handled (like rated) so it disappears
            video_marked = False
            if video_name:
                try:
                    manual_ratings = self._load_json(self.manual_ratings_file, {})
                    if video_name not in manual_ratings:
                        manual_ratings[video_name] = {
                            "rating": {
                                "banned": True,
                                "overall_quality": 2,
                                "visual_quality": 2,
                                "motion_quality": 2,
                                "prompt_adherence": 2,
                                "creativity": 2,
                                "technical_quality": 2,
                                "comments": "Auto-banned via QA UI"
                            },
                            "timestamp": time.time(),
                            "rated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        self._save_json(self.manual_ratings_file, manual_ratings)
                        video_marked = True
                except Exception:
                    pass

            # Append ban history
            self._append_ban_history({
                'timestamp': time.time(),
                'video_name': video_name,
                'combo_key': combo_key,
                'params': params,
            })

            self.send_json_response({"status": "success", "banned_combo": combo_key, "total_banned": len(banned), "video_marked": video_marked})
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)})

    def serve_qa_page(self):
        html = """
<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QA Ban Console</title>
  <style>
    body { font-family: Arial, sans-serif; background: #0b1e39; color: #fff; margin: 0; padding: 20px; }
    .nav { display:flex; gap:10px; margin-bottom:12px; }
    .nav button { background:#3949AB; color:#fff; border:0; padding:8px 12px; border-radius:6px; cursor:pointer; }
    .nav button:hover { background:#303F9F; }
    .card { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; padding: 16px; margin: 10px 0; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    button { background: #ff6b6b; color: #fff; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; }
    button:hover { background: #ff4040; }
    .small { opacity: 0.8; font-size: 0.9rem; }
  </style>
  <script src="/static/qa_console.js"></script>
</head>
<body>
  <div class="nav">
    <button onclick="location.href='/'">Головна</button>
    <button onclick="location.href='/search'">Пошук</button>
    <button onclick="location.href='/qa'">QA</button>
  </div>
  <h2>QA Ban Console</h2>
  <p class="small">Використовуйте, щоб забанити комбінацію параметрів за назвою відео. Список нижче — невідоцінені відео.</p>
  <div id="list">Завантаження...</div>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_main_page_with_ban(self):
        # Wrapper page with ban toolbar over base UI
        wrapper = """
<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Enhanced Review + Ban</title>
  <style>
    .ban-toolbar { position: fixed; top: 10px; right: 10px; z-index: 9999; }
    .ban-toolbar button { background:#ff6b6b; color:#fff; border:none; padding:8px 12px; border-radius:6px; cursor:pointer; }
    .nav { position: fixed; top: 10px; left: 10px; z-index: 9999; display:flex; gap:8px; }
    .nav button { background:#3949AB; color:#fff; border:0; padding:8px 12px; border-radius:6px; cursor:pointer; }
    .nav button:hover { background:#303F9F; }
  </style>
</head>
<body>
  <iframe id="base" src="/base" style="position:fixed; inset:0; width:100%; height:100%; border:0;"></iframe>
  <div class="nav">
    <button onclick="location.href='/'">Головна</button>
    <button onclick="location.href='/search'">Пошук</button>
    <button onclick="location.href='/qa'">QA</button>
  </div>
  <div class="ban-toolbar">
    <button onclick="banCurrent()">🚫 Ban combo (current video)</button>
  </div>
  <script src="/static/ban_wrapper.js"></script>
</body>
</html>
"""
        if urllib.parse.urlparse(self.path).path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(wrapper.encode('utf-8'))
        else:
            return super().do_GET()

    def serve_videos_api_qa(self):
        """API like base, but enrich with auto_metrics/combo from review_queue.json"""
        url_parts = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(url_parts.query)
        offset = int(query_params.get('offset', ['0'])[0])
        limit = int(query_params.get('limit', ['50'])[0])

        manual_ratings = self._load_json(self.manual_ratings_file, {})
        rated_videos = set(manual_ratings.keys())

        video_files = glob.glob(f"{self.video_dir}*.mp4")
        video_files.sort(key=os.path.getmtime, reverse=True)

        knowledge = self._load_json(self.knowledge_file, {"history": []})
        review_queue = self._load_json(self.review_queue_file, {"pending": []})
        rq_map = {}
        try:
            for entry in review_queue.get("pending", []):
                name = os.path.basename(entry.get("original_path", ""))
                if name:
                    rq_map[name] = entry
        except Exception:
            pass

        unrated_videos = []
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            if video_name not in rated_videos:
                unrated_videos.append(video_path)

        paginated_videos = unrated_videos[offset:offset + limit]

        result = []
        for video_path in paginated_videos:
            video_name = os.path.basename(video_path)
            stat = os.stat(video_path)

            video_details, match_info = self._enhanced_video_search(video_name, knowledge)

            rq = rq_map.get(video_name)
            # Merge enhanced metrics from knowledge (metrics) with basic auto_metrics
            auto_metrics = {}
            if isinstance(video_details, dict):
                try:
                    md = video_details.get('metrics') or {}
                    if isinstance(md, dict):
                        auto_metrics.update(md)
                except Exception:
                    pass
                try:
                    am = video_details.get('auto_metrics') or {}
                    if isinstance(am, dict):
                        auto_metrics.update(am)
                except Exception:
                    pass
            if (not auto_metrics) and rq and isinstance(rq.get('auto_metrics'), dict):
                auto_metrics = rq.get('auto_metrics')

            combo = video_details.get('combo') if isinstance(video_details, dict) else None
            if (not combo) and rq and isinstance(rq.get('combo'), list):
                combo = rq.get('combo')

            video_info = {
                'name': video_name,
                'size_mb': round(stat.st_size / 1024 / 1024, 1),
                'created': time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                'prompt': video_details.get('prompt', 'Промпт недоступний') if isinstance(video_details, dict) else 'Промпт недоступний',
                'fps': (video_details.get('params', {}) or {}).get('fps', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'width': (video_details.get('params', {}) or {}).get('width', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'height': (video_details.get('params', {}) or {}).get('height', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'seconds': (video_details.get('params', {}) or {}).get('seconds', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'seed': (video_details.get('params', {}) or {}).get('seed', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'combo': combo or 'N/A',
                'auto_metrics': auto_metrics or {},
                'iteration': video_details.get('iteration', 'N/A') if isinstance(video_details, dict) else 'N/A',
                'found_in_knowledge': match_info['found'],
                'match_method': match_info['method'],
                'search_details': match_info['details']
            }
            result.append(video_info)

        self.send_json_response(result)

    def serve_search_api(self):
        """Search across files + rated + knowledge params."""
        url_parts = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url_parts.query)

        name_sub = (q.get('q', [''])[0] or '').lower()
        rated_filter = (q.get('rated', ['all'])[0] or 'all')  # 'all'|'true'|'false'
        banned_filter = q.get('banned', [''])[0]
        ref_filter = q.get('reference', [''])[0]
        min_score = float(q.get('min_score', ['-1'])[0] or -1)
        min_overall = float(q.get('min_overall', ['-1'])[0] or -1)
        fps_filter = q.get('fps', [''])[0]
        width_filter = q.get('width', [''])[0]
        height_filter = q.get('height', [''])[0]
        sampler_filter = (q.get('sampler', [''])[0] or '').lower()
        scheduler_filter = (q.get('scheduler', [''])[0] or '').lower()
        offset = int(q.get('offset', ['0'])[0] or 0)
        limit = int(q.get('limit', ['100'])[0] or 100)

        manual = self._load_json(self.manual_ratings_file, {})
        knowledge = self._load_json(self.knowledge_file, {"history": []})

        # Build candidate name set: files + rated keys
        files = [os.path.basename(p) for p in glob.glob(f"{self.video_dir}*.mp4")]
        names = set(files) | set(manual.keys())

        def knowledge_entry_for(name: str):
            det, _mi = self._enhanced_video_search(name, knowledge)
            return det or {}

        def get_score(det: dict):
            if not isinstance(det, dict):
                return None
            if 'overall' in det:
                return det.get('overall')
            m = det.get('metrics', {}) or {}
            for k in ('overall', 'overall_simple', 'blended_overall'):
                if k in m:
                    return m.get(k)
            return None

        def params_of(det: dict):
            p = (det.get('params', {}) if isinstance(det, dict) else {}) or {}
            return {
                'fps': p.get('fps'), 'width': p.get('width'), 'height': p.get('height'),
                'sampler': p.get('sampler') or (det.get('combo', [None, None])[0] if isinstance(det.get('combo'), list) else None),
                'scheduler': p.get('scheduler') or (det.get('combo', [None, None])[1] if isinstance(det.get('combo'), list) else None),
            }

        results = []
        for name in names:
            det = knowledge_entry_for(name)
            score = get_score(det)
            p = params_of(det)
            mr = manual.get(name)
            is_rated = bool(mr)
            rating = (mr or {}).get('rating', {})
            is_banned = bool(rating.get('banned'))
            is_ref = bool(rating.get('is_reference'))
            overall_manual = rating.get('overall_quality')

            # Filters
            if name_sub and name_sub not in name.lower():
                continue
            if rated_filter == 'true' and not is_rated:
                continue
            if rated_filter == 'false' and is_rated:
                continue
            if banned_filter:
                want = banned_filter.lower() == 'true'
                if is_banned != want:
                    continue
            if ref_filter:
                want = ref_filter.lower() == 'true'
                if is_ref != want:
                    continue
            if min_score >= 0 and (score is None or float(score) < min_score):
                continue
            if min_overall >= 0 and (overall_manual is None or float(overall_manual) < min_overall):
                continue
            if fps_filter and str(p.get('fps')) != fps_filter:
                continue
            if width_filter and str(p.get('width')) != width_filter:
                continue
            if height_filter and str(p.get('height')) != height_filter:
                continue
            if sampler_filter and str(p.get('sampler') or '').lower() != sampler_filter:
                continue
            if scheduler_filter and str(p.get('scheduler') or '').lower() != scheduler_filter:
                continue

            results.append({
                'name': name,
                'score': score,
                'manual_overall': overall_manual,
                'rated': is_rated,
                'banned': is_banned,
                'reference': is_ref,
                'params': p,
                'exists': name in files
            })

        results.sort(key=lambda x: (x['rated'] is False, -(x['score'] or -1)))
        paged = results[offset:offset + limit]
        self.send_json_response({'total': len(results), 'items': paged})

    def serve_video_details_api(self):
        url_parts = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url_parts.query)
        name = q.get('name', [''])[0]
        if not name:
            return self.send_json_response({'status': 'error', 'message': 'name required'})

        manual = self._load_json(self.manual_ratings_file, {})
        knowledge = self._load_json(self.knowledge_file, {"history": []})
        det, mi = self._enhanced_video_search(name, knowledge)
        mr = manual.get(name, {})
        info = {
            'name': name,
            'found_in_knowledge': bool(mi.get('found')),
            'details': det or {},
            'manual_rating': mr.get('rating') if isinstance(mr, dict) else None,
            'exists': os.path.exists(os.path.join(self.video_dir, name))
        }
        try:
            rq = self._load_json(self.review_queue_file, {"pending": []})
            for entry in rq.get("pending", []):
                en = os.path.basename(entry.get("original_path", ""))
                if en == name:
                    d = info['details'] if isinstance(info['details'], dict) else {}
                    if isinstance(entry.get('auto_metrics'), dict):
                        d.setdefault('auto_metrics', {}).update(entry['auto_metrics'])
                    if isinstance(entry.get('params'), dict):
                        d.setdefault('params', {}).update(entry['params'])
                    if isinstance(entry.get('combo'), list):
                        d.setdefault('combo', entry['combo'])
                    info['details'] = d
                    break
        except Exception:
            pass
        self.send_json_response(info)

    def serve_search_page(self):
        html = """
<!DOCTYPE html>
<html lang="uk">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Пошук відео</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b1e39; color:#fff; margin:0; padding:20px; }
    .nav { display:flex; gap:10px; margin-bottom:12px; }
    .nav button { background:#3949AB; color:#fff; border:0; padding:8px 12px; border-radius:6px; cursor:pointer; }
    .nav button:hover { background:#303F9F; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
    .card { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; padding: 12px; margin: 10px 0; }
    input, select { padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.3); background:rgba(0,0,0,0.2); color:#fff; }
    button { padding:8px 12px; border:0; background:#2196F3; color:#fff; border-radius:6px; cursor:pointer; }
    button:hover { background:#1976D2; }
    table { width:100%; border-collapse:collapse; margin-top:12px; }
    th, td { padding:8px; border-bottom:1px solid rgba(255,255,255,0.2); text-align:left; }
    .pill { padding:2px 6px; border-radius:10px; background:rgba(255,255,255,0.15); font-size:12px; }
    a { color:#90caf9; }
  </style>
</head>
<body>
  <div class="nav">
    <button onclick="location.href='/'">Головна</button>
    <button onclick="location.href='/search'">Пошук</button>
    <button onclick="location.href='/qa'">QA</button>
  </div>
  <h2>Пошук відео</h2>
  <div class="row">
    <input id="q" placeholder="Назва містить..." />
    <select id="rated"><option value="all">Всі</option><option value="true">Тільки оцінені</option><option value="false">Тільки неоцінені</option></select>
    <select id="banned"><option value="">Бан? (всі)</option><option value="true">Тільки бан</option><option value="false">Без бану</option></select>
    <select id="reference"><option value="">Еталон? (всі)</option><option value="true">Так</option><option value="false">Ні</option></select>
    <input id="min_score" type="number" step="0.01" placeholder="min score (0..1)" />
    <input id="min_overall" type="number" step="1" placeholder="min overall (1..10)" />
    <input id="fps" type="number" placeholder="fps" style="width:80px" />
    <input id="width" type="number" placeholder="w" style="width:80px" />
    <input id="height" type="number" placeholder="h" style="width:80px" />
    <input id="sampler" placeholder="sampler" style="width:120px" />
    <input id="scheduler" placeholder="scheduler" style="width:120px" />
    <button onclick="runSearch()">Шукати</button>
  </div>

  <table id="res"><thead><tr><th>Відео</th><th>Score</th><th>Manual</th><th>Статус</th><th>Параметри</th><th></th></tr></thead><tbody></tbody></table>

  <script src="/static/search_page.js"></script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def serve_watch_page(self):
        url_parts = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url_parts.query)
        name = q.get('name', [''])[0]
        if not name:
            name = ''
        html = """
<!DOCTYPE html>
<html lang=\"uk\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Перегляд відео</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b1e39; color:#fff; margin:0; padding:20px; }
    .nav { display:flex; gap:10px; margin-bottom:12px; }
    .nav a, .nav button { background:#3949AB; color:#fff; border:0; padding:8px 12px; border-radius:6px; cursor:pointer; text-decoration:none; display:inline-block; }
    .nav a:hover, .nav button:hover { background:#303F9F; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
    .card { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; padding: 12px; }
    .video-wrap { max-width: 960px; }
    video { width: 100%; max-height: 480px; border-radius:8px; }
    input, textarea { width:100%; padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.3); background:rgba(0,0,0,0.2); color:#fff; }
    label { display:block; margin-top:8px; }
    button { padding:8px 12px; border:0; background:#4CAF50; color:#fff; border-radius:6px; cursor:pointer; margin-top:8px; }
  </style>
</head>
<body>
  <div class=\"nav\">
    <a href=\"/\">Головна</a>
    <a href=\"/search\">Пошук</a>
    <a href=\"/qa\">QA</a>
  </div>
  <h2 id=\"title\">Перегляд відео</h2>
  <div class=\"grid\">
    <div class=\"card video-wrap\">
      <div id=\"video_box\"></div>
    </div>
    <div class=\"card\">
      <h3>Поточні дані</h3>
      <pre id=\"details\" style=\"white-space:pre-wrap\"></pre>
      <h3>Оцінка</h3>
      <form id=\"rate_form\"> 
        <input type=\"hidden\" name=\"video_name\" value=\"\"> 
        <label>Overall <input name=\"overall_quality\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label>Visual <input name=\"visual_quality\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label>Motion <input name=\"motion_quality\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label>Prompt adherence <input name=\"prompt_adherence\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label>Creativity <input name=\"creativity\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label>Technical <input name=\"technical_quality\" type=\"number\" min=\"1\" max=\"10\"></label>
        <label><input type=\"checkbox\" name=\"is_reference\"> Позначити як еталон</label>
        <label>Коментарі<textarea name=\"comments\"></textarea></label>
        <button type=\"submit\">Зберегти</button>
      </form>
      <div id=\"msg\"></div>
    </div>
  </div>

  <script src=\"/static/watch_page.js\"></script>
</body>
</html>
"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

if __name__ == '__main__':
    # Порт з ENV (SERVER_PORT/PORT), дефолт 8189
    PORT = int(os.environ.get('SERVER_PORT', os.environ.get('PORT', '8189')))

    print("🚀 Запуск Enhanced Video Review System (Quick Fix) для RunPod...")
    print(f"📁 Робоча директорія: {os.environ.get('WAN22_SYSTEM_DIR', '/workspace/wan22_system/')}")
    print(f"🎬 Директорія відео: {os.environ.get('COMFY_OUTPUT_DIR', '/workspace/ComfyUI/output/')}")
    print(f"💾 JSON файли: {os.path.join(os.environ.get('WAN22_SYSTEM_DIR', '/workspace/wan22_system/'), 'auto_state')}")
    print(f"🌐 Сервер запущено на порту {PORT}")
    print(f"📱 Відкрийте в RunPod URL для порту {PORT}")

    print("\n🔧 Quick Fix виправлення:")
    print("  ✅ Виправлена обробка URL параметрів для /api/videos")
    print("  ✅ Додане детальне логування для діагностики")
    print("  ✅ Спрощений алгоритм пошуку відео")
    print("  ✅ Покращена обробка помилок")

    # Дозволяємо перевикористання адреси та багатопоточність для кращої продуктивності
    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    with ThreadingTCPServer(("", PORT), QAReviewHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Сервер зупинено")