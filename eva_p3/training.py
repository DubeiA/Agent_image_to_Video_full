
def main():
    """Main function to demonstrate Enhanced Video Agent v4."""

    print("Enhanced Video Agent v4 - Improved")
    print("=" * 50)

    # Create configuration
    config = AnalysisConfig(
        frame_skip=3,
        max_frames=200,
        face_confidence_threshold=0.8,
        anatomical_threshold=0.7,
        face_quality_threshold=0.8,
        artifact_threshold=0.6,
        enable_temporal_analysis=True,
        enable_frequency_analysis=True,
        enable_optical_flow=True,
        gpu_enabled=True,
        log_level="INFO",
        save_detailed_logs=True,
        log_training_process=True
    )

    # Initialize Enhanced Video Agent
    agent = EnhancedVideoAgent(config)

    try:
        # Example: Train models (with synthetic data for demonstration)
        print("\nInitializing training system...")
        training_data = {
            'anatomical_data': {'epochs': 5},
            'face_quality_data': {'epochs': 5},
            'artifact_data': {'epochs': 5},
            'ensemble_epochs': 3
        }

        training_results = agent.train_models(training_data)
        print(f"Training completed. Models trained: {len(training_results)}")

        # Example: Analyze video (replace with actual video path)
        video_path = "sample_video.mp4"
        if os.path.exists(video_path):
            print(f"\nAnalyzing video: {video_path}")
            result = agent.analyze_video(video_path)

            print(f"Analysis Result:")
            print(f"  - Is Deepfake: {result.is_deepfake}")
            print(f"  - Confidence: {result.confidence:.4f}")
            print(f"  - Processing Time: {result.processing_time:.2f}s")
            print(f"  - Frames Analyzed: {result.frames_analyzed}")
            print(f"  - Faces Detected: {result.faces_detected}")
            print(f"  - Anatomical Score: {result.anatomical_score:.4f}")
            print(f"  - Face Quality Score: {result.face_quality_score:.4f}")
            print(f"  - Artifact Score: {result.artifact_score:.4f}")
        else:
            print(f"Video file not found: {video_path}")
            print("Skipping video analysis demonstration")

        # Generate reports
        print("\nGenerating final reports...")
        performance_report = agent.generate_performance_report()
        print(f"Performance report generated with {performance_report['performance_metrics']['total_videos_processed']} videos processed")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")

    finally:
        # Cleanup resources
        print("\nCleaning up resources...")
        agent.cleanup_resources()
        print("Enhanced Video Agent v4 execution completed")



# ==== END: Improved classes ====
