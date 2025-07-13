import nltk
from nltk.corpus import stopwords

# Step 1: Download stopwords
nltk.download('stopwords')

# Step 2: Test if it works
print("✅ Sample stopwords loaded from NLTK:")
print(stopwords.words('english')[:10])  # Show first 10 stopwords
