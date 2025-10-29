import os
import json
import re
from pathlib import Path
from collections import defaultdict
import math

# Library untuk IR
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import scoring
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class IRSystem:
    def __init__(self):
        self.index_dir = "indexdir"
        self.datasets_dir = "datasets"
        self.stopwords = self.load_stopwords()
        self.documents = []
        self.vectorizer = None
        self.doc_vectors = None
        self.ix = None
        
    def load_stopwords(self):
        """Load Indonesian stopwords"""
        stopwords = {
            'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 
            'dia', 'dua', 'ia', 'seperti', 'jika', 'sehingga', 'kembali',
            'dan', 'di', 'dari', 'ini', 'itu', 'dengan', 'tidak', 'ada', 'adalah',
            'sebagai', 'oleh', 'saat', 'dapat', 'sudah', 'saya', 'akan', 'atau',
            'telah', 'dalam', 'bahwa', 'mereka', 'karena', 'tersebut', 'bisa',
            'hanya', 'hal', 'juga', 'serta', 'hingga', 'harus', 'lebih', 'bagi',
            'sebuah', 'satu', 'suatu', 'masih', 'terhadap', 'setiap', 'belum',
            'kami', 'ketika', 'pun', 'sangat', 'paling', 'agar', 'kita'
        }
        return stopwords
    
    def preprocess_text(self, text):
        """
        Preprocessing text:
        - Case folding
        - Tokenization
        - Stopword removal
        """
        if pd.isna(text) or text is None:
            return ''
        
        # Convert to string
        text = str(text)
        
        # Case folding
        text = text.lower()
        
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenization
        tokens = text.split()
        
        # Stopword removal
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_documents_from_csv(self):
        """Load documents from CSV files"""
        print("\n[INFO] Loading documents from CSV files...")
        
        csv_files = {
            'etd-usk': 'etd_usk.csv',
            'etd-ugm': 'etd_ugm.csv',
            'kompas': 'kompas.csv',
            'tempo': 'tempo.csv',
            'mojok': 'mojok.csv'
        }
        
        total_docs = 0
        
        for dataset_name, csv_file in csv_files.items():
            csv_path = os.path.join(self.datasets_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"[WARNING] File {csv_file} not found. Skipping...")
                continue
            
            try:
                # Read CSV
                df = pd.read_csv(csv_path, encoding='utf-8')
                
                # Check if CSV has required columns
                if df.empty:
                    print(f"[WARNING] {csv_file} is empty. Skipping...")
                    continue
                
                # Try different possible column names for content
                content_columns = ['content', 'text', 'abstract', 'abstrak', 'isi', 'body', 'description']
                content_col = None
                
                for col in content_columns:
                    if col in df.columns:
                        content_col = col
                        break
                
                if content_col is None:
                    # Use the last column as content
                    content_col = df.columns[-1]
                    print(f"[INFO] Using column '{content_col}' as content for {dataset_name}")
                
                # Process each row
                for idx, row in df.iterrows():
                    content = row[content_col]
                    
                    if pd.isna(content) or str(content).strip() == '':
                        continue
                    
                    # Create document
                    doc = {
                        'id': f"{dataset_name}_{idx}",
                        'source': dataset_name,
                        'filename': f"doc_{idx}",
                        'content': str(content),
                        'preprocessed': self.preprocess_text(content)
                    }
                    
                    # Add title if exists
                    title_columns = ['title', 'judul', 'headline']
                    for title_col in title_columns:
                        if title_col in df.columns and not pd.isna(row[title_col]):
                            doc['title'] = str(row[title_col])
                            break
                    
                    self.documents.append(doc)
                    total_docs += 1
                
                print(f"[INFO] Loaded {len(df)} documents from {csv_file}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {csv_file}: {str(e)}")
        
        print(f"[SUCCESS] Total {total_docs} documents loaded.\n")
        return total_docs > 0
    
    def load_documents_from_folders(self):
        """Load documents from folder structure (original method)"""
        print("\n[INFO] Loading documents from folders...")
        
        datasets = ['etd-usk', 'etd-ugm', 'kompas', 'tempo', 'mojok']
        total_docs = 0
        
        for dataset in datasets:
            dataset_path = os.path.join(self.datasets_dir, dataset)
            
            if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
                continue
            
            files = [f for f in os.listdir(dataset_path) if f.endswith('.txt')]
            
            for file in files:
                file_path = os.path.join(dataset_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        doc = {
                            'id': f"{dataset}_{file}",
                            'source': dataset,
                            'filename': file,
                            'content': content,
                            'preprocessed': self.preprocess_text(content)
                        }
                        self.documents.append(doc)
                        total_docs += 1
                except Exception as e:
                    print(f"[ERROR] Failed to load {file}: {str(e)}")
            
            if len(files) > 0:
                print(f"[INFO] Loaded {len(files)} documents from {dataset}")
        
        return total_docs > 0
    
    def load_documents(self):
        """Load documents - try CSV first, then folders"""
        # Try loading from CSV files first
        if self.load_documents_from_csv():
            return True
        
        # If no CSV files found, try folder structure
        print("[INFO] No CSV files found. Trying folder structure...")
        return self.load_documents_from_folders()
    
    def build_index(self):
        """Build Whoosh index"""
        print("[INFO] Building Whoosh index...")
        print(f"[INFO] Processing {len(self.documents)} documents... Please wait...")
        
        # Create schema
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            source=TEXT(stored=True),
            filename=TEXT(stored=True),
            content=TEXT(stored=True),
            preprocessed=TEXT(stored=True)
        )
        
        # Create index directory
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
        
        # Create index
        self.ix = create_in(self.index_dir, schema)
        writer = self.ix.writer()
        
        # Add documents to index with progress
        total = len(self.documents)
        for i, doc in enumerate(self.documents, 1):
            writer.add_document(
                doc_id=doc['id'],
                source=doc['source'],
                filename=doc['filename'],
                content=doc['content'],
                preprocessed=doc['preprocessed']
            )
            
            # Show progress every 5000 documents
            if i % 5000 == 0:
                print(f"  [PROGRESS] Indexed {i}/{total} documents ({i*100//total}%)")
        
        print(f"  [PROGRESS] Committing index... This may take a while...")
        writer.commit()
        print(f"[SUCCESS] Index created with {len(self.documents)} documents.\n")
    
    def build_bow_representation(self):
        """Build Bag of Words representation using CountVectorizer"""
        print("[INFO] Building Bag of Words representation...")
        
        # Get preprocessed texts
        texts = [doc['preprocessed'] for doc in self.documents]
        
        # Create CountVectorizer
        self.vectorizer = CountVectorizer(max_features=1000)
        
        # Fit and transform
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"[SUCCESS] BoW created with vocabulary size: {len(self.vectorizer.vocabulary_)}\n")
    
    def search_query(self, query_text, top_k=5):
        """Search query using Whoosh and rank using Cosine Similarity"""
        print(f"\n[SEARCH] Query: '{query_text}'")
        print("="*60)
        
        # Preprocess query
        preprocessed_query = self.preprocess_text(query_text)
        print(f"[INFO] Preprocessed query: '{preprocessed_query}'\n")
        
        if not preprocessed_query:
            print("[WARNING] Query becomes empty after preprocessing. Try different terms.\n")
            return []
        
        # Method 1: Whoosh search (for initial filtering)
        with self.ix.searcher(weighting=scoring.TF_IDF()) as searcher:
            query = QueryParser("preprocessed", self.ix.schema).parse(preprocessed_query)
            results = searcher.search(query, limit=50)
            
            if len(results) == 0:
                print("[INFO] No documents found matching the query.\n")
                return []
            
            matched_doc_ids = [hit['doc_id'] for hit in results]
        
        # Method 2: Cosine Similarity ranking
        query_vector = self.vectorizer.transform([preprocessed_query])
        
        # Get indices of matched documents
        matched_indices = [i for i, doc in enumerate(self.documents) if doc['id'] in matched_doc_ids]
        
        if not matched_indices:
            print("[INFO] No documents found in BoW representation.\n")
            return []
        
        # Calculate cosine similarity
        matched_doc_vectors = self.doc_vectors[matched_indices]
        similarities = cosine_similarity(query_vector, matched_doc_vectors)[0]
        
        # Combine doc indices with similarity scores
        doc_scores = list(zip(matched_indices, similarities))
        
        # Sort by similarity score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K results
        top_results = doc_scores[:top_k]
        
        # Display results
        print(f"[RESULTS] Top {len(top_results)} documents:\n")
        
        results_list = []
        for rank, (doc_idx, score) in enumerate(top_results, 1):
            doc = self.documents[doc_idx]
            
            print(f"Rank {rank} | Score: {score:.4f}")
            print(f"Source: {doc['source']}")
            print(f"File: {doc['filename']}")
            
            # Show title if exists
            if 'title' in doc:
                print(f"Title: {doc['title']}")
            
            # Show snippet
            snippet = doc['content'][:200].replace('\n', ' ')
            print(f"Snippet: {snippet}...")
            print("-" * 60)
            
            results_list.append({
                'rank': rank,
                'score': score,
                'source': doc['source'],
                'filename': doc['filename'],
                'content': doc['content']
            })
        
        print()
        return results_list
    
    def load_existing_index(self):
        """Load existing index if available"""
        if os.path.exists(self.index_dir) and os.listdir(self.index_dir):
            print("[INFO] Loading existing index...")
            try:
                self.ix = open_dir(self.index_dir)
                
                # Load documents metadata
                with self.ix.searcher() as searcher:
                    for doc in searcher.documents():
                        self.documents.append({
                            'id': doc['doc_id'],
                            'source': doc['source'],
                            'filename': doc['filename'],
                            'content': doc['content'],
                            'preprocessed': doc['preprocessed']
                        })
                
                # Rebuild BoW representation
                self.build_bow_representation()
                
                print(f"[SUCCESS] Loaded {len(self.documents)} documents from existing index.\n")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load existing index: {str(e)}")
                return False
        return False


def main():
    """Main CLI program"""
    ir_system = IRSystem()
    index_loaded = False
    
    print("\n" + "="*60)
    print("     INFORMATION RETRIEVAL SYSTEM")
    print("     UTS Praktikum Penelusuran Informasi")
    print("="*60)
    
    while True:
        print("\n=== INFORMATION RETRIEVAL SYSTEM ===")
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("====================================")
        
        choice = input("\nPilih menu: ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("LOAD & INDEX DATASET")
            print("="*60)
            
            # Try to load existing index first
            if ir_system.load_existing_index():
                index_loaded = True
            else:
                # Load and build new index
                if ir_system.load_documents():
                    ir_system.build_index()
                    ir_system.build_bow_representation()
                    index_loaded = True
                else:
                    print("[ERROR] Failed to load documents. Please check datasets folder.")
        
        elif choice == '2':
            if not index_loaded:
                print("\n[WARNING] Please load and index dataset first (Menu 1).\n")
                continue
            
            print("\n" + "="*60)
            print("SEARCH QUERY")
            print("="*60)
            
            query = input("\nMasukkan query pencarian: ").strip()
            
            if query:
                ir_system.search_query(query)
            else:
                print("[WARNING] Query tidak boleh kosong.\n")
        
        elif choice == '3':
            print("\n[INFO] Terima kasih telah menggunakan sistem IR!")
            print("="*60 + "\n")
            break
        
        else:
            print("\n[ERROR] Pilihan tidak valid. Silakan pilih 1, 2, atau 3.\n")


if __name__ == "__main__":
    main()