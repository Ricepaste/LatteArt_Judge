# main/download_voc.py
import argparse
import os
import torchvision

def main():
    parser = argparse.ArgumentParser(description="Download and extract Pascal VOC 2012 Semantic Segmentation Dataset")
    parser.add_argument("--output_dir", type=str, default="main/data/voc", help="Directory to save the dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print(f"🚀 Downloading Pascal VOC 2012 Dataset to: {args.output_dir}")
    print("This will download and extract the dataset (~2GB download, ~2.6GB uncompressed).")
    print("=" * 60)

    try:
        # download=True will fetch the dataset and extract it automatically
        torchvision.datasets.VOCSegmentation(
            root=args.output_dir,
            year="2012",
            image_set="train",
            download=True
        )
        print("\n✅ Dataset successfully downloaded and extracted!")
        print(f"Verify the directory structure: {args.output_dir}/VOCdevkit/VOC2012/")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")

if __name__ == "__main__":
    main()
