from LINKS.CP import make_empty_submission, evaluate_submission
file_name = "merged_submission.npy"

if evaluate_submission is not None:
    print("\nEvaluating submission with provided helper...")
    print(f"File Name:{file_name}")
    hv = evaluate_submission(
        submission= file_name,
        target_curves="target_curves.npy"
    )
    print(f"Average hypervolume (helper): {hv}")
else:
    print("\nNOTE: LINKS.CP.evaluate_submission not found; skipping helper scoring.")