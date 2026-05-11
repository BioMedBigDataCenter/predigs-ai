from rdflib import Graph, URIRef, RDFS, OWL
from collections import defaultdict
import pandas as pd
import os
import scanpy as sc
import pandas as pd
#####################################################################
###################Cell Ontology Hierarchy Path Construction#########
#####################################################################
# Load OWL ontology
g = Graph()
g.parse("E:\\cl.owl", format="xml")  # Replace with your OWL file path

# Set root node URI (e.g., CL:0000000 is "cell")
ROOT = URIRef("http://purl.obolibrary.org/obo/CL_0000000")

# Build RDF graph as subclass -> parent mapping
subclass_dict = defaultdict(list)
for s, p, o in g.triples((None, RDFS.subClassOf, None)):
    if isinstance(s, URIRef) and isinstance(o, URIRef):
        subclass_dict[s].append(o)

# Optional: map URI -> name using label
label_dict = {}
for s, _, o in g.triples((None, RDFS.label, None)):
    label_dict[s] = str(o)

# Find URI of the target class by label
def find_class_uri_by_label(label):
    for s, l in label_dict.items():
        if l.lower() == label.lower():
            return s
    return None

# Recursively build all paths: from target class up to ROOT
def get_all_paths(current, path=None):
    if path is None:
        path = [current]
    if current == ROOT or current not in subclass_dict:
        return [path]
    paths = []
    for parent in subclass_dict[current]:
        paths.extend(get_all_paths(parent, path + [parent]))
    return paths

# Read CSV, cell_type column name is 'cell_type'
df_input = pd.read_csv("E://statistics//CL-20250506.csv")
rows = []
for target_label in df_input['cell_type']:
    target_uri = find_class_uri_by_label(target_label)
    if not target_uri:
        print(f"Cannot find URI for label: {target_label}")
    else:
        all_paths = get_all_paths(target_uri)
        # Take the shortest path as the representative path
        shortest_path = min(all_paths, key=len)
        shortest_path_sequence = '->'.join([label_dict.get(cls, str(cls)) for cls in reversed(shortest_path)])
        print("✅ Representative path: " + shortest_path_sequence)
        # Get string representation of each path and concatenate them
        path_sequences = []
        for path in all_paths:
            # Get string representation of each path
            path_sequence = '->'.join([label_dict.get(cls, str(cls)) for cls in reversed(path)])
            path_sequences.append(path_sequence)
        # Join all path strings using " |---PATH---| "
        all_paths_joined = ' |---PATH---| '.join(path_sequences)
        # Add the result to row
        rows.append({
            "target_uri": target_uri,
            "target_label": target_label,
            "all_paths": all_paths_joined,
            "shortest_path": shortest_path_sequence
        })

# Convert results to DataFrame
df_output = pd.DataFrame(rows)
# Output to CSV file
df_output.to_csv("E://statistics//CL_path_summary.csv", index=False)
# Optional: display first few rows to confirm output
print(df_output.head())

####################################################################
##Manual Curation and Construction of Modified Cell Ontology Paths##
df_output = pd.read_csv("E://statistics//CL_path_summary_curated.csv", index=False)
####################################################################

#####################################################################
###################Cell Ontology Hierarchy Path Mapping##############
#####################################################################
cl_id_map = dict(zip(df_output['target_label'], df_output['target_uri']))
all_paths_map = dict(zip(df_output['target_label'], df_output['all_paths']))
level_seq_map = dict(zip(df_output['target_label'], df_output['shortest_path']))

# Set the folder path containing multiple h5ad files
data_dir = "./"  # Replace with your own path
all_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
total_files = len(all_files)

# Iterate through all .h5ad files in the directory
for i, file in enumerate(all_files, 1):
    file_path = os.path.join(data_dir, file)
    adata = sc.read_h5ad(file_path)
    print(f"--- Processing file: {file} ({i}/{total_files}) ---")
    # Add cell name as a column (index is originally cell name)
    obs_df = adata.obs.copy()
    adata.obs['CL_id'] = adata.obs['cell_type'].map(cl_id_map)
    adata.obs['all_paths'] = adata.obs['cell_type'].map(all_paths_map)
    adata.obs['level_sequence'] = adata.obs['cell_type'].map(level_seq_map)
    unmatched = adata.obs['cell_type'][adata.obs['CL_id'].isna()].unique()
    if len(unmatched) > 0:
        print(f"Warning: {len(unmatched)} cell types not found in df_output: {unmatched}")
    adata.write(file_path)
