import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")

from src.data_processing import process_data
from src.train_baseline import train_baseline_model
from src.train_boosted import train_boosted_model
from src.evaluation import run_evaluation
from src.final_test_evaluation import run_final_test_evaluation
from src.prefix_analysis import run_prefix_analysis
from src.segment_analysis import run_segment_analysis


def main():
    print("=" * 60)
    print("Step 1/7: Processing raw data")
    print("=" * 60)
    process_data(save=True)

    print("\n" + "=" * 60)
    print("Step 2/7: Training baseline model")
    print("=" * 60)
    train_baseline_model()

    print("\n" + "=" * 60)
    print("Step 3/7: Training boosted model")
    print("=" * 60)
    train_boosted_model()

    print("\n" + "=" * 60)
    print("Step 4/7: Validation evaluation and model selection")
    print("=" * 60)
    run_evaluation()

    print("\n" + "=" * 60)
    print("Step 5/7: Final held-out test evaluation")
    print("=" * 60)
    run_final_test_evaluation()

    print("\n" + "=" * 60)
    print("Step 6/7: Prefix-based validation analysis")
    print("=" * 60)
    run_prefix_analysis()

    print("\n" + "=" * 60)
    print("Step 7/7: Segment-based validation analysis")
    print("=" * 60)
    run_segment_analysis()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()