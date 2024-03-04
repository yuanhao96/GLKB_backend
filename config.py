# database
cypher_file = '/var/www/glkb/KGBackend/utils/cypher.json' # '/var/www/glkb/KGBackend/utils/cypher_alpha.json'
glkb_uri = "neo4j://localhost:8687" # "neo4j://localhost:7687"
vocab_graph_projection = "vocab_subgraph"

# port
debug_port = 8100
port = 8000

# enriched sets
gene_set = '/var/www/glkb/KGBackend/graph_stats/Genes and Gene Products_neighbor_sets.json'
chemical_set = '/var/www/glkb/KGBackend/graph_stats/Chemicals and Drugs_neighbor_sets.json'
disease_set = '/var/www/glkb/KGBackend/graph_stats/Diseases_neighbor_sets.json'
organism_set = '/var/www/glkb/KGBackend/graph_stats/Organisms_neighbor_sets.json'
anatomy_set = '/var/www/glkb/KGBackend/graph_stats/Anatomy_neighbor_sets.json'
pathway_set = '/var/www/glkb/KGBackend/graph_stats/Pathway_neighbor_sets.json'
go_set = '/var/www/glkb/KGBackend/graph_stats/GO_neighbor_sets.json'
high_freq_set = '/var/www/glkb/KGBackend/graph_stats/high_freq_neighbor_sets.json'
high_freq_set_feats = '/var/www/glkb/KGBackend/graph_stats/high_freq_feats.json'

# biocypher ver
# gene_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/Gene_neighbor_sets.json'
# chemical_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/ChemicalEntity_neighbor_sets.json'
# disease_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/DiseaseOrPhenotypicFeature_neighbor_sets.json'
# organism_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/OrganismTaxon_neighbor_sets.json'
# anatomy_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/AnatomicalEntity_neighbor_sets.json'
# pathway_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/Pathway_neighbor_sets.json'
# go_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/BiologicalProcessOrActivity_neighbor_sets.json'
# high_freq_set = '/var/www/glkb/KGBackend/graph_stats/biocypher/high_freq_neighbor_sets.json'
# high_freq_set_feats = '/var/www/glkb/KGBackend/graph_stats/biocypher/high_freq_feats.json'

# article stats
pmid2date = '/var/www/glkb/KGBackend/graph_stats/pmid2date.json'
pmid2ncit = '/var/www/glkb/KGBackend/graph_stats/pmid2ncitation.json'

# embeddings
semantic_embedding = '/nfs/turbo/umms-drjieliu/proj/medlineKG/data/glkb_processed_data/hgt_semantic_embeddings.pk'