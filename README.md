# 📚 AI-Powered Book Recommendation System

An intelligent book recommendation system that leverages machine learning, natural language processing, and vector similarity search to help users discover their next great read. Built with Python, this system combines data analysis, text classification, and semantic search capabilities.

## 🌟 Features

- **Semantic Search**: Find books using natural language queries like "a book about medieval wars" or "something to teach children about nature"
- **Interactive Web Interface**: User-friendly Gradio dashboard for easy book discovery
- **Data Analysis**: Comprehensive exploratory data analysis of book metadata and ratings
- **Text Classification**: Automated categorization of books using transformer models
- **Vector Embeddings**: Uses sentence transformers for semantic understanding of book descriptions
- **Large Dataset**: Works with 7,000+ books with rich metadata

## 🔧 Technology Stack

- **Python 3.x**
- **Machine Learning**: 
  - Transformers (Hugging Face)
  - Sentence Transformers
  - ChromaDB for vector storage
- **Data Processing**: 
  - Pandas
  - NumPy
- **Visualization**: 
  - Matplotlib
  - Seaborn
- **Web Interface**: 
  - Gradio
- **Environment Management**: 
  - python-dotenv
- **Data Source**: 
  - Kaggle Hub

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ahmed-BenRejeb/Book_recommendations.git
   cd Book_recommendations
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn
   pip install transformers sentence-transformers
   pip install chromadb langchain langchain-community langchain-chroma
   pip install gradio kagglehub
   pip install python-dotenv
   ```

3. **Set up environment variables** (if needed)
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Usage

#### 1. Data Exploration and Processing
Run the Jupyter notebooks to understand the data and process it:

```bash
jupyter notebook data-exploration.ipynb
```

This notebook will:
- Download the book dataset from Kaggle
- Perform data cleaning and filtering
- Generate visualizations and correlation analysis
- Create the cleaned dataset (`books_cleaned.csv`)

#### 2. Text Classification
```bash
jupyter notebook text-classification.ipynb
```

This notebook demonstrates:
- Zero-shot classification of books into categories
- Using transformer models for automated categorization

#### 3. Vector Search Setup
```bash
jupyter notebook vector-search.ipynb
```

This notebook shows:
- Creating embeddings for book descriptions
- Setting up ChromaDB for vector storage
- Implementing semantic search functionality

#### 4. Launch the Web Interface
```bash
python gradio-dashboard.py
```

The Gradio interface will be available at `http://localhost:7860` where you can:
- Enter natural language queries about books you're looking for
- Adjust the number of recommendations
- Get detailed book information with descriptions

## 📁 Project Structure

```
Book_recommendations/
├── 📊 data-exploration.ipynb      # Data analysis and preprocessing
├── 🏷️ text-classification.ipynb   # Book categorization
├── 🔍 vector-search.ipynb         # Semantic search implementation
├── 🖥️ gradio-dashboard.py         # Web interface
├── 📋 books_cleaned.csv           # Processed book dataset
├── 📄 tagged_description.txt      # Text data for vector search
├── 🔧 .env                        # Environment variables
├── 📝 README.md                   # Project documentation
└── 🚫 .gitignore                  # Git ignore rules
```

## 🔍 How It Works

### Data Processing Pipeline
1. **Data Acquisition**: Downloads book dataset from Kaggle (7k books with metadata)
2. **Data Cleaning**: Filters books with sufficient description length (25+ words)
3. **Feature Engineering**: Creates combined title-subtitle fields and tagged descriptions
4. **Quality Control**: Removes entries with missing critical information

### Recommendation Engine
1. **Text Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` to create vector embeddings
2. **Vector Storage**: Stores embeddings in ChromaDB for efficient similarity search
3. **Query Processing**: Converts user queries into embeddings
4. **Similarity Matching**: Finds most similar books using cosine similarity
5. **Result Ranking**: Returns top-k most relevant books with metadata

### Classification System
- Uses Facebook's BART model for zero-shot classification
- Categorizes books into simplified categories (Fiction, Nonfiction, Children's books)
- Supports custom category mapping for better organization

## 📈 Dataset Information

The system works with a comprehensive book dataset containing:
- **7,000+ books** with rich metadata
- **Book Information**: Title, authors, ISBN, publication year
- **Reader Engagement**: Average ratings, number of ratings
- **Content Details**: Descriptions, categories, page counts
- **Visual Elements**: Thumbnail images

## 🎯 Example Queries

The system understands natural language queries like:
- "A book to teach children about nature"
- "Something about medieval wars and battles"
- "Romance novels with strong female characters"
- "Science fiction books about space exploration"
- "Self-help books about productivity"

## 🛠️ Customization

### Adding New Categories
Modify the `category_mapping` dictionary in the text classification notebook to add or change book categories.

### Adjusting Search Parameters
- Change the embedding model in the vector search setup
- Modify the similarity threshold for recommendations
- Adjust the number of results returned

### Interface Customization
The Gradio interface can be customized by modifying `gradio-dashboard.py`:
- Change the appearance and layout
- Add new input fields or filters
- Modify the output format

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Kaggle** for providing the comprehensive book dataset
- **Hugging Face** for the transformer models and sentence transformers
- **ChromaDB** team for the vector database solution
- **Gradio** team for the intuitive web interface framework

## 📧 Contact

**Ahmed Ben Rejeb**
- GitHub: [@Ahmed-BenRejeb](https://github.com/Ahmed-BenRejeb)

---

⭐ **Star this repository if you found it helpful!**