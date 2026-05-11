import scanpy as sc
import os
import glob


def read_mapping_from_txt(txt_file):
    """Read mapping relationships from a txt file."""
    mapping_dict = {}
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Skip possible header lines
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or '\t' not in line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                raw_name = parts[0].strip()
                new_name = parts[1].strip()
                mapping_dict[raw_name] = new_name
        print(f"Loaded {len(mapping_dict)} mapping relations from {txt_file}")
        # Print first few mapping examples for inspection
        print("First 5 mapping examples:")
        for i, (k, v) in enumerate(list(mapping_dict.items())[:5]):
            print(f"  {k} -> {v}")
    except FileNotFoundError:
        print(f"Error: Mapping file {txt_file} not found")
        return None
    except Exception as e:
        print(f"Error reading mapping file: {str(e)}")
        return None
    return mapping_dict

def process_h5ad_files(mapping_txt_file):
    # Read mapping relations from txt file
    celltype_mapping = read_mapping_from_txt(mapping_txt_file)
    if celltype_mapping is None:
        print("Unable to read mapping relations, exiting")
        return
    # Create output directory
    output_dir = "./v2"
    os.makedirs(output_dir, exist_ok=True)
    # Get all h5ad files in the current directory
    h5ad_files = glob.glob("*.h5ad")
    if not h5ad_files:
        print("No .h5ad files found in current directory")
        return
    print(f"\nFound {len(h5ad_files)} .h5ad files")
    for i, file_path in enumerate(h5ad_files):
        try:
            print(f"\nProcessing file {i+1}/{len(h5ad_files)}: {file_path}")
            # Read h5ad file
            adata = sc.read_h5ad(file_path)
            # Check if 'basic_celltype' column exists
            if 'basic_celltype' not in adata.obs.columns:
                print(f"Warning: File {file_path} does not have 'basic_celltype' column, skipping")
                continue
            # Backup original column (convert to string to avoid categorical issues)
            if 'basic_celltype_original' not in adata.obs.columns:
                adata.obs['basic_celltype_original'] = adata.obs['basic_celltype'].astype(str).copy()
            # Get list of original cell types
            original_celltypes = adata.obs['basic_celltype'].astype(str)
            original_types = original_celltypes.unique()
            # Perform cell type mapping replacement
            mapped_celltypes = original_celltypes.map(
                lambda x: celltype_mapping.get(x, x)
            )
            # Assign mapped results back to adata.obs
            adata.obs['basic_celltype'] = mapped_celltypes
            # Count mapping changes (ensure both columns are string type for comparison)
            original_str = adata.obs['basic_celltype_original'].astype(str)
            mapped_str = adata.obs['basic_celltype'].astype(str)
            mapped_count = (original_str != mapped_str).sum()
            total_count = len(adata.obs)
            print(f"  Total cells: {total_count}, Mapped: {mapped_count}")
            # Count unmapped cell types
            mapped_types = adata.obs['basic_celltype'].unique()
            unmapped_original = [t for t in original_types if t not in celltype_mapping]
            print(f"  Total cells: {total_count}")
            print(f"  Mapped cells: {mapped_count}")
            print(f"  Number of original cell types: {len(original_types)}")
            print(original_types)
            print(f"  Number of mapped cell types: {len(mapped_types)}")
            print(mapped_types)
            if unmapped_original:
                print(f"  Unmapped cell types ({len(unmapped_original)}): {unmapped_original[:10]}")  # show only first 10
            # Build output file path
            output_path = os.path.join(output_dir, file_path)
            if 'basic_celltype_original' in adata.obs.columns:
                adata.obs = adata.obs.drop(columns=['basic_celltype_original'])
            # Save processed file
            adata.write_h5ad(output_path)
            print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue

def main():
    # Mapping file name (can be modified as needed)
    mapping_files = ["meta_celltype.txt"]
    # Find mapping file
    mapping_file = None
    for mf in mapping_files:
        if os.path.exists(mf):
            mapping_file = mf
            break
    if mapping_file is None:
        print("Mapping file not found. Please ensure one of the following files exists in current directory:")
        for mf in mapping_files:
            print(f"  - {mf}")
        print("\nOr manually specify the mapping file path:")
        mapping_file = input("Enter mapping file path: ").strip()
        
        if not os.path.exists(mapping_file):
            print(f"Error: File {mapping_file} does not exist")
            return
    print(f"Using mapping file: {mapping_file}")
    process_h5ad_files(mapping_file)

if __name__ == "__main__":
    main()