import os
import numpy as np
import gc
import time
import warnings
from sklearn.ensemble import IsolationForest

from large_scale_utils import (
    load_data,
    inject_extreme_anomalies,
    split_data,
    preprocess,
    random_search_if,
    evaluate_model,
    measure_time_memory,
    ensemble_iforest_parallel,
    save_results,
    extract_features,
    train_iforest_in_chunks,
    aggregate_models_scores,
    aggregate_models_preds,
    calibrate_threshold,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = "results/large_scale"
DATA_PATH = "datasets/5m_dataset.csv"


def run_experiment(random_state: int = None):
   """
    Execute the large-scale Isolation Forest evaluation.

    Compares serial and parallel model implementations across
    preprocessing, training, detection, and evaluation phases.

    Tracks execution time, memory consumption, detection metrics,
    and performance speedup before saving the results.
    """
  
    if random_state is None:
        random_state = np.random.randint(0, 10000)

    print("=== LARGE-SCALE EVALUATION START ===")

    # PHASE 1: DATA PREPROCESSING
    print("\n[Phase 1: Data Loading and Preprocessing...]")
    try:
        (df, prep_time, prep_mem, _) = measure_time_memory(load_data, DATA_PATH)
    except MemoryError:
        raise SystemExit("[ERROR] Dataset too large for available memory. Try sampling smaller portion.")

    # Inject synthetic extreme anomalies and extract relevant features
    df = inject_extreme_anomalies(df, rate=0.02, random_state=random_state)
    df = extract_features(df)

    # Divide dataset into training, validation, and testing subsets
    train_df, val_df, test_df = split_data(df)

    # Release the original dataframe to reduce memory consumption
    del df
    gc.collect()

    (preproc_result, preproc_time, preproc_mem, _) = measure_time_memory(preprocess, train_df, val_df, test_df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preproc_result

    # Convert to float32 to reduce memory
    X_train = X_train.astype(np.float32, copy=False)
    X_val = X_val.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    del train_df, val_df, test_df
    gc.collect()

    # HYPERPARAMETER TUNING (RANDOM SEARCH) - Search for the best Isolation Forest configuration
    print("\n[Tuning hyperparameters with random search...]")
    best_model, best_params = random_search_if(X_train, X_val, y_val, n_iter=20, random_state=random_state)
    print("Best parameters found:", best_params)

    # Ensure best_params exists
    if best_params is None:
        raise SystemExit("[ERROR] Random search failed to find any params.")

    # THRESHOLD CALIBRATION
    '''
    - Retrain the selected configuration using the full training dataset to obtain more representative anomaly scores for calibration.
    - Determine the decision threshold using validation data.
    - Release validation arrays after threshold calibration
    '''

    print("\n[Calibrating anomaly score threshold...]")
    try:
        full_model = IsolationForest(**best_params, n_jobs=1, random_state=random_state)
        full_model.fit(X_train)
    except Exception as e:
        print("[WARN] Retraining full model for calibration failed; falling back to best_model from random search.", e)
        full_model = best_model

    best_threshold = calibrate_threshold(full_model, X_val, y_val)
    print(f"[INFO] Calibrated threshold: {best_threshold:.6f}")

    del X_val, y_val
    gc.collect()

    # PHASE 2: MODEL TRAINING
    '''
    - Train the baseline Isolation Forest using a single CPU.
    - Configure safer parameters for memory-efficient parallel training.
    - Divide training data into chunks for parallel model training.
    - Normalize memory measurements for consistent reporting.
    '''
    print("\n[Phase 2: Training Phase]")

    # SERIAL MODEL (1 CPU) trained on full X_train
    serial_model = IsolationForest(**best_params, n_jobs=1, random_state=random_state)
    print("[Training Serial Model...]")
    (_, train_time_serial, mem_serial, _) = measure_time_memory(serial_model.fit, X_train)
    gc.collect()

    safe_params = best_params.copy()
    safe_params["bootstrap"] = False
    safe_params["max_samples"] = min(best_params.get("max_samples", 0.1), 0.05)
    # in case n_estimators large, reduce per-chunk later inside train_iforest_in_chunks

    # CHUNKED SAFE PARALLEL TRAINING (returns list of models)
    print("[Training Parallel Model using safe chunked training...]")
    n_chunks = max(2, (os.cpu_count() or 2) // 2) 
    try:
        (models_parallel, train_time_parallel, mem_parallel, _) = measure_time_memory(
            train_iforest_in_chunks, X_train, safe_params, n_chunks, random_state
        )
    except Exception as e:
        print("[WARNING] Chunked training failed:", e)
        print("[FALLBACK] Training a single reduced parallel_model with safe settings...")
        safe_params_2 = safe_params.copy()
        safe_params_2["n_estimators"] = min(safe_params_2.get("n_estimators", 100), 30)
        safe_params_2["max_samples"] = min(safe_params_2.get("max_samples", 0.05), 0.02)
        safe_params_2["bootstrap"] = False
        fallback_model = IsolationForest(**safe_params_2, n_jobs=1, random_state=random_state)
        (_, train_time_parallel, mem_parallel, _) = measure_time_memory(fallback_model.fit, X_train)
        models_parallel = [fallback_model]

    gc.collect()

    mem_serial = abs(mem_serial) if mem_serial is not None else 0.0
    mem_parallel = abs(mem_parallel) if mem_parallel is not None else 0.0

    # PHASE 3: ANOMALY DETECTION
    '''
    - Generate anomaly scores using the serial ensemble configuration
    - Aggregate anomaly scores from the chunk-trained parallel models
    - Use ensemble-based detection if no parallel models are available
    - Measure prediction thresholding time independently
    '''

    print("\n[Phase 3: Detection Phase]")

    preds_serial_ens, scores_serial = ensemble_iforest_parallel(
        X_train, X_test, best_params, n_models=3, agg="mean", return_scores=True
    )

    if isinstance(models_parallel, list) and len(models_parallel) > 0:
        scores_parallel = aggregate_models_scores(models_parallel, X_test, agg="mean")
        preds_parallel_ens = (scores_parallel < best_threshold).astype(int)
    else:
        preds_parallel_ens, scores_parallel = ensemble_iforest_parallel(
            X_train, X_test, safe_params, n_models=3, mode="parallel", agg="mean", return_scores=True
        )
        preds_parallel_ens = (scores_parallel < best_threshold).astype(int)

    start = time.time()
    preds_serial_final = (scores_serial < best_threshold).astype(int)
    detect_time_serial = time.time() - start

    start = time.time()
    preds_parallel_final = (scores_parallel < best_threshold).astype(int)
    detect_time_parallel = time.time() - start

    gc.collect()

    # PHASE 4: EVALUATION 
    print("\n[Phase 4: Evaluation Phase]")

    # Adapter class that exposes aggregated scores through a model-like interface
    class ChunkedIFProxy:
        def __init__(self, models):
            self.models = models

        def score_samples(self, X):
            return aggregate_models_scores(self.models, X, agg="mean")

    parallel_model_proxy = ChunkedIFProxy(models_parallel)

    # Evaluate serial model performance using the calibrated threshold
    (eval_result_serial, eval_time_serial, eval_mem_serial, _) = measure_time_memory(
        evaluate_model, serial_model, X_test, y_test, best_threshold
    )

    # Evaluate aggregated parallel model performance
    (eval_result_parallel, eval_time_parallel, eval_mem_parallel, _) = measure_time_memory(
        evaluate_model, parallel_model_proxy, X_test, y_test, best_threshold
    )

    p_serial, r_serial, f_serial = eval_result_serial
    p_parallel, r_parallel, f_parallel = eval_result_parallel

    # PERFORMANCE REPORT
'''
- Display execution time for each experimental phase
- Display peak memory consumption for measured operations
- Calculate training and detection speedup ratios
- Identify which phase gained more from parallelization
'''

    print("\n=== RESULTS (All-CPU Parallel Mode, Memory Safe) ===")
    print(f"Serial:    Precision={p_serial:.4f} Recall={r_serial:.4f} F1={f_serial:.4f}")
    print(f"Parallel:  Precision={p_parallel:.4f} Recall={r_parallel:.4f} F1={f_parallel:.4f}")
    print(f"Calibrated Threshold: {best_threshold:.6f}")

    print("\nExecution Time (s):")
    print(f" Preprocessing:   {preproc_time:.3f}")
    print(f" Train Serial:    {train_time_serial:.3f}")
    print(f" Train Parallel:  {train_time_parallel:.3f}")
    print(f" Detect Serial:   {detect_time_serial:.3f}")
    print(f" Detect Parallel: {detect_time_parallel:.3f}")
    print(f" Eval Serial:     {eval_time_serial:.3f}")
    print(f" Eval Parallel:   {eval_time_parallel:.3f}")

    print("\nMemory Usage (MB):")
    print(f" Preprocess peak: {preproc_mem:.2f}")
    print(f" Train Serial:    {mem_serial:.2f}")
    print(f" Train Parallel:  {mem_parallel:.2f}")
    print(f" Eval Serial:     {eval_mem_serial:.2f}")
    print(f" Eval Parallel:   {eval_mem_parallel:.2f}")

    train_speedup = train_time_serial / train_time_parallel if train_time_parallel > 0 else np.nan
    detect_speedup = detect_time_serial / detect_time_parallel if detect_time_parallel > 0 else np.nan

    benefit = (
        "Training" if train_speedup > detect_speedup
        else "Detection" if detect_speedup > train_speedup
        else "Both"
    )

    print(f"\n>>> {benefit} phase benefited most from parallelization.")
    print(f"Training Speedup: {train_speedup:.2f}x")
    print(f"Detection Speedup: {detect_speedup:.2f}x")

    # SAVE EXPERIMENTAL RESULTS
    results = {
        "best_params": best_params,
        "best_threshold": float(best_threshold),
        "serial": {"precision": p_serial, "recall": r_serial, "f1": f_serial},
        "parallel": {"precision": p_parallel, "recall": r_parallel, "f1": f_parallel},
        "speedup": {"train_speedup": train_speedup, "detect_speedup": detect_speedup, "benefit": benefit},
        "phases": {
            "preprocessing": {"time_s": preproc_time, "peak_memory_mb": preproc_mem},
            "training_serial": {"time_s": train_time_serial, "peak_memory_mb": mem_serial},
            "training_parallel": {"time_s": train_time_parallel, "peak_memory_mb": mem_parallel},
            "detection_serial": {"time_s": detect_time_serial},
            "detection_parallel": {"time_s": detect_time_parallel},
            "evaluation_serial": {"time_s": eval_time_serial, "peak_memory_mb": eval_mem_serial},
            "evaluation_parallel": {"time_s": eval_time_parallel, "peak_memory_mb": eval_mem_parallel},
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(os.path.join(RESULTS_DIR, "large_scale_results.json"), results)

    print("\nResults saved to:", os.path.join(RESULTS_DIR, "large_scale_results.json"))
    print("=== LARGE-SCALE EVALUATION COMPLETE ===")


if __name__ == "__main__":
    run_experiment()
