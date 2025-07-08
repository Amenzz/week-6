📄 EDA Summary Report
The dataset contains consumer complaints across various financial products. Our initial analysis revealed that products such as credit cards and money transfers attract a high volume of complaints. After filtering to include only the five relevant product categories, we retained a significant portion of the data.

A closer look at complaint narratives revealed a wide variation in word count. While many complaints are concise, a substantial number are lengthy and detailed. We found over 100,000 complaints with narratives, and cleaned the text for improved downstream processing. This cleaned dataset is now ready for use in semantic embedding and retrieval tasks for the RAG system.
📄 Report Section 
Chunking Strategy:
We used a sliding window chunking approach with a chunk_size of 100 words and a chunk_overlap of 20. This balance ensures each chunk retains semantic coherence while capturing context from neighboring sections, which improves downstream retrieval performance.

Embedding Model Choice:
We selected sentence-transformers/all-MiniLM-L6-v2 for its efficiency and solid performance in semantic similarity tasks. It provides a good trade-off between speed and accuracy, making it ideal for real-time search scenarios like this one.