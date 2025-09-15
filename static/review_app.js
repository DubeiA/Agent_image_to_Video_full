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
                    <div class="info-title">📝 Промпт</div>
                    <div class="info-content">${video.prompt && video.prompt !== 'Промпт недоступний' ? video.prompt : '⚠️ Промпт не знайдений в knowledge.json'}</div>
                </div>
                
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


