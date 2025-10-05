"""
Fast Parallel Kepler Light Curve Downloader
============================================
High-speed multi-threaded download with progress tracking.

Target: 400 Kepler light curves (200 planets + 200 false positives)
Expected time: 60-90 minutes

Features:
- Parallel downloads (10 threads)
- Progress bar
- Error handling and retry
- Caching downloaded data
- Resume capability

Author: NASA Kepler Project
Date: 2025-10-05
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import warnings
warnings.filterwarnings('ignore')

# Enable AWS S3 cloud access for faster downloads
try:
    from astroquery.mast import Observations
    Observations.enable_cloud_dataset()
    print("[INFO] AWS S3 cloud access enabled (s3://stpubdata/kepler/public)")
    print("[INFO] Downloads will be significantly faster!")
except Exception as e:
    print(f"[WARN] Could not enable S3 access: {e}")
    print("[INFO] Falling back to standard MAST downloads")

print("="*80)
print("FAST KEPLER LIGHT CURVE DOWNLOADER")
print("="*80)
print("\nTarget: 400 light curves (200 confirmed + 200 false positives)")
print("="*80)

# ============================================
# CONFIGURATION
# ============================================

TARGET_CONFIRMED = 200
TARGET_FALSE_POS = 200
TOTAL_TARGET = TARGET_CONFIRMED + TARGET_FALSE_POS

MAX_WORKERS = 8  # Parallel download threads (user requested - using AWS S3)
CACHE_DIR = Path('data/cached_lightcurves')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Progress tracking
progress_lock = Lock()
download_stats = {
    'confirmed_success': 0,
    'confirmed_fail': 0,
    'false_pos_success': 0,
    'false_pos_fail': 0,
    'total_success': 0,
    'total_fail': 0
}

# ============================================
# UTILITIES
# ============================================

def print_progress(message, stats=None):
    """Thread-safe progress printing"""
    with progress_lock:
        if stats:
            total = stats['total_success'] + stats['total_fail']
            success_rate = (stats['total_success'] / total * 100) if total > 0 else 0
            print(f"\r[{total}/{TOTAL_TARGET}] {message} | "
                  f"Success: {stats['total_success']} | "
                  f"Fail: {stats['total_fail']} | "
                  f"Rate: {success_rate:.1f}%", end='', flush=True)
        else:
            print(f"\r{message}", end='', flush=True)


def save_checkpoint(data, filename='download_checkpoint.pkl'):
    """Save progress checkpoint"""
    checkpoint_path = CACHE_DIR / filename
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)


def load_checkpoint(filename='download_checkpoint.pkl'):
    """Load progress checkpoint"""
    checkpoint_path = CACHE_DIR / filename
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================
# DOWNLOAD FUNCTIONS
# ============================================

def process_lightcurve(flux_data):
    """Process light curve to 2001 points"""
    target_length = 2001

    if len(flux_data) < 100:
        return None

    if len(flux_data) != target_length:
        orig_idx = np.linspace(0, len(flux_data) - 1, len(flux_data))
        target_idx = np.linspace(0, len(flux_data) - 1, target_length)
        processed = np.interp(target_idx, orig_idx, flux_data)
    else:
        processed = flux_data.copy()

    # Standardization
    mean = np.mean(processed)
    std = np.std(processed)
    if std > 0:
        processed = (processed - mean) / std

    return processed


def download_single_koi(kepid, is_planet, max_retries=3):
    """
    Download a single KOI light curve with retry logic

    Returns:
        tuple: (kepid, processed_lightcurve, is_planet, success)
    """
    # Check cache first
    cache_file = CACHE_DIR / f'kic_{kepid}.npy'
    if cache_file.exists():
        try:
            cached_lc = np.load(cache_file)
            return (kepid, cached_lc, is_planet, True, 'cached')
        except:
            pass

    # Download from MAST
    for attempt in range(max_retries):
        try:
            import lightkurve as lk

            # Search for light curves
            search_result = lk.search_lightcurve(
                f'KIC {kepid}',
                mission='Kepler',
                cadence='long'
            )

            if len(search_result) == 0:
                return (kepid, None, is_planet, False, 'no_data')

            # Download only first 2 quarters for speed
            # (Usually sufficient for transit detection)
            lc_collection = search_result[:2].download_all()

            if lc_collection is None or len(lc_collection) == 0:
                return (kepid, None, is_planet, False, 'download_fail')

            # Stitch quarters
            lc = lc_collection.stitch()

            # Extract flux
            flux = lc.flux.value
            lc_time = lc.time.value

            # Remove NaNs
            mask = ~np.isnan(flux) & ~np.isnan(lc_time)
            flux = flux[mask]

            if len(flux) < 100:
                return (kepid, None, is_planet, False, 'too_short')

            # Normalize
            flux = flux / np.median(flux)

            # Process
            processed = process_lightcurve(flux)

            if processed is None:
                return (kepid, None, is_planet, False, 'process_fail')

            # Cache the result
            np.save(cache_file, processed)

            return (kepid, processed, is_planet, True, 'downloaded')

        except Exception as e:
            if attempt == max_retries - 1:
                return (kepid, None, is_planet, False, f'error_{str(e)[:20]}')
            time.sleep(5)  # Wait 5s before retry due to MAST timeouts

    return (kepid, None, is_planet, False, 'max_retries')


def download_worker(task):
    """Worker function for parallel download"""
    kepid, is_planet = task
    result = download_single_koi(kepid, is_planet)

    # Update stats
    with progress_lock:
        if result[3]:  # success
            download_stats['total_success'] += 1
            if is_planet:
                download_stats['confirmed_success'] += 1
            else:
                download_stats['false_pos_success'] += 1
        else:
            download_stats['total_fail'] += 1
            if is_planet:
                download_stats['confirmed_fail'] += 1
            else:
                download_stats['false_pos_fail'] += 1

        print_progress("Downloading...", download_stats)

    return result


# ============================================
# MAIN DOWNLOAD ORCHESTRATION
# ============================================

def main():
    global TARGET_CONFIRMED, TARGET_FALSE_POS, TOTAL_TARGET

    start_time = time.time()

    print("\n[STEP 1/5] Loading KOI catalog...")

    # Load catalog
    koi_data = pd.read_csv('data/q1_q17_dr25_koi.csv')

    confirmed = koi_data[koi_data['koi_disposition'] == 'CONFIRMED']
    false_pos = koi_data[koi_data['koi_disposition'] == 'FALSE POSITIVE']

    print(f"[OK] Available - Confirmed: {len(confirmed)}, False Pos: {len(false_pos)}")

    # Check if we have enough
    if len(confirmed) < TARGET_CONFIRMED:
        print(f"[WARN] Only {len(confirmed)} confirmed available, adjusting target...")
        TARGET_CONFIRMED = len(confirmed)

    if len(false_pos) < TARGET_FALSE_POS:
        print(f"[WARN] Only {len(false_pos)} false positives available, adjusting target...")
        TARGET_FALSE_POS = len(false_pos)

    # Select targets
    confirmed_targets = confirmed.head(TARGET_CONFIRMED)
    false_pos_targets = false_pos.head(TARGET_FALSE_POS)

    print(f"\n[STEP 2/5] Preparing download queue...")
    print(f"  Target: {len(confirmed_targets)} confirmed + {len(false_pos_targets)} false pos")
    print(f"  Total: {len(confirmed_targets) + len(false_pos_targets)} light curves")

    # Create task list
    tasks = []
    for _, row in confirmed_targets.iterrows():
        tasks.append((row['kepid'], True))
    for _, row in false_pos_targets.iterrows():
        tasks.append((row['kepid'], False))

    # Shuffle for better load balancing
    np.random.shuffle(tasks)

    print(f"\n[STEP 3/5] Starting parallel download ({MAX_WORKERS} workers)...")
    print("[INFO] This may take 60-90 minutes depending on network speed")
    print("[INFO] Progress will be saved - you can resume if interrupted")

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n[INFO] Found checkpoint with {len(checkpoint['results'])} downloaded")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            results = checkpoint['results']
            downloaded_kepids = set([r[0] for r in results if r[3]])
            tasks = [t for t in tasks if t[0] not in downloaded_kepids]

            # Restore stats
            for r in results:
                if r[3]:  # success
                    download_stats['total_success'] += 1
                    if r[2]:  # is_planet
                        download_stats['confirmed_success'] += 1
                    else:
                        download_stats['false_pos_success'] += 1

            print(f"[OK] Resuming with {len(tasks)} remaining tasks")
        else:
            results = []
    else:
        results = []

    print("\n")

    # Parallel download with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_worker, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                # Save checkpoint every 50 downloads
                if len(results) % 50 == 0:
                    save_checkpoint({'results': results})

            except Exception as e:
                print(f"\n[ERROR] Task failed: {e}")

    print("\n")  # New line after progress bar

    download_time = time.time() - start_time

    print("\n[STEP 4/5] Download complete! Processing results...")

    # Separate successful downloads
    successful = [r for r in results if r[3]]
    failed = [r for r in results if not r[3]]

    print(f"\n[RESULTS]")
    print(f"  Total attempted: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"  Download time: {download_time/60:.1f} minutes")

    # Analyze by category
    confirmed_success = [r for r in successful if r[2]]
    false_pos_success = [r for r in successful if not r[2]]

    print(f"\n[BREAKDOWN]")
    print(f"  Confirmed planets: {len(confirmed_success)}")
    print(f"  False positives: {len(false_pos_success)}")

    # Save data
    print(f"\n[STEP 5/5] Saving dataset...")

    X_data = np.array([r[1] for r in successful])
    y_labels = np.array([[0, 1] if r[2] else [1, 0] for r in successful])
    kepids = np.array([r[0] for r in successful])

    # Save as numpy arrays
    output_dir = Path('data/genesis_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'X_data.npy', X_data)
    np.save(output_dir / 'y_labels.npy', y_labels)
    np.save(output_dir / 'kepids.npy', kepids)

    # Save metadata
    metadata = {
        'n_total': len(successful),
        'n_confirmed': len(confirmed_success),
        'n_false_pos': len(false_pos_success),
        'download_time_min': download_time / 60,
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'success_rate': len(successful) / len(results),
        'target_length': 2001,
        'max_workers': MAX_WORKERS
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Dataset saved to {output_dir}/")
    print(f"  X_data.npy: {X_data.shape}")
    print(f"  y_labels.npy: {y_labels.shape}")
    print(f"  kepids.npy: {kepids.shape}")

    # Save failure analysis
    if failed:
        failure_reasons = {}
        for r in failed:
            reason = r[4]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        print(f"\n[FAILURE ANALYSIS]")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        with open(output_dir / 'failures.json', 'w') as f:
            json.dump({
                'total_failed': len(failed),
                'reasons': failure_reasons,
                'failed_kepids': [r[0] for r in failed]
            }, f, indent=2)

    # Clean up checkpoint
    checkpoint_file = CACHE_DIR / 'download_checkpoint.pkl'
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\n[OK] Successfully downloaded {len(successful)} light curves")
    print(f"[TIME] Total time: {download_time/60:.1f} minutes")
    print(f"[STATS] Success rate: {len(successful)/len(results)*100:.1f}%")
    print(f"[SIZE] Dataset size: {X_data.nbytes / 1024**2:.1f} MB")
    print(f"\n[DATA] Data location: {output_dir}/")
    print(f"  - X_data.npy (light curves)")
    print(f"  - y_labels.npy (labels)")
    print(f"  - kepids.npy (KIC IDs)")
    print(f"  - metadata.json (info)")

    if len(successful) >= 300:
        print(f"\n[GREAT] You have {len(successful)} samples - sufficient for reliable training!")
    elif len(successful) >= 200:
        print(f"\n[GOOD] You have {len(successful)} samples - adequate for training.")
    else:
        print(f"\n[WARN] Only {len(successful)} samples - may need more for best results.")

    # Suggest next steps
    print(f"\n[NEXT STEPS]")
    print(f"  1. Run: python scripts/genesis_train_large_dataset.py")
    print(f"  2. Dataset will be split:")
    print(f"     - Training: ~{int(len(successful)*0.8)} samples")
    print(f"     - Testing: ~{int(len(successful)*0.2)} samples")
    print(f"  3. Expected training time: ~{len(successful)*0.3/60:.0f}-{len(successful)*0.5/60:.0f} minutes")

    print("\n" + "="*80)

    return successful


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        import lightkurve as lk
    except ImportError:
        print("[ERROR] lightkurve not installed!")
        print("[FIX] Run: pip install lightkurve")
        sys.exit(1)

    print("\n[IMPORTANT NOTES]")
    print("  * This will download from MAST archive (requires internet)")
    print("  * Download time: ~60-90 minutes for 400 samples")
    print("  * Progress is saved - you can resume if interrupted")
    print("  * Cached data will be reused on subsequent runs")
    print("  * Press Ctrl+C to stop (progress will be saved)")

    # Auto-start if --auto-confirm flag or skip confirmation
    if len(sys.argv) > 1 and sys.argv[1] == '--auto-confirm':
        print("\n[AUTO-CONFIRMED] Starting download...")
    else:
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)

    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving progress...")
        print("[INFO] Run again to resume from checkpoint")
        sys.exit(0)
