#!/usr/bin/env python3
"""Test script to verify HuggingFace checkpoint path parsing."""

def parse_hf_path(checkpoint_path):
    """Test the HuggingFace path parsing logic."""
    import os
    
    if "/" in checkpoint_path and not os.path.exists(checkpoint_path):
        # Parse HuggingFace path
        if "huggingface.co/" in checkpoint_path:
            # Extract from full URL
            parts = checkpoint_path.split("huggingface.co/")[1].split("/")
            repo_id = f"{parts[0]}/{parts[1]}"
            # Find filename after 'blob/main/' or 'resolve/main/'
            if "blob" in parts or "resolve" in parts:
                try:
                    idx = parts.index("main") + 1 if "main" in parts else parts.index("blob") + 2
                    filename = "/".join(parts[idx:])
                except (ValueError, IndexError):
                    filename = parts[-1]
            else:
                filename = parts[-1]
        elif checkpoint_path.count("/") >= 2:
            # Format: "username/repo/filename"
            parts = checkpoint_path.split("/", 2)
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = parts[2]
        else:
            raise ValueError(f"HuggingFace path must include filename: {checkpoint_path}")
        
        return repo_id, filename
    return None, None


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "https://huggingface.co/alphaXiv/trm-model-maze/blob/main/maze_hard_step_32550",
        "alphaXiv/trm-model-maze/maze_hard_step_32550",
        "https://huggingface.co/alphaXiv/trm-model-maze/resolve/main/maze_hard_step_32550",
    ]
    
    print("Testing HuggingFace path parsing:\n")
    for path in test_cases:
        try:
            repo_id, filename = parse_hf_path(path)
            if repo_id:
                print(f"✓ Input:    {path}")
                print(f"  Repo ID:  {repo_id}")
                print(f"  Filename: {filename}")
                print()
        except Exception as e:
            print(f"✗ Input: {path}")
            print(f"  Error: {e}")
            print()
