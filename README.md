# Text-based Graph Convolutional Network with tensorflow 1.x

This project was designed to create a text based gcn in Tensorflow 1.x. Each node in the graph represents either a word or a document from the corpus. So the edges are relative relations between either word-word pairs or word-document pairs. 

For the demo I used the IMBD mood data set for a semi-supervised training.


## Sources:
- https://arxiv.org/pdf/1809.05679.pdf
- https://arxiv.org/pdf/1609.02907.pdf
- https://github.com/plkmo/Bible_Text_GCN



# Dependencies
 Python (3.5 +), networkx (2.4), tensorflow(1.12.0),


# Calculate the Graph Edges
****

![]("Notebooks/weights_formula.png")

**PMI**  

* PMI is the *Point-wise Mutual Information* between pairs of co-occurring words over a sliding window $\#W$ that we fix to be of 10-words length
* $\#W(i)$ is the number of sliding windows in a corpus that contain word $i$  
* $\#W(i,j)$ is the number of sliding windows that contain both word $i$ and $j$  
* $\#W$ is the total number of sliding windows in the corpus


**TF-iDF** 
For the word-document relationship in the graph the author proposed the tf-idf method.
Our target is to create a data frame with the dimension [num docs x vocabulary size] containing the TF-iDF score for each word and document.


# Semi-Supervised Node Classification

Our model $f(X, A)$ is a two layer convolution network with sprectral convolutions. The model is flexible in terms of it relies just on the feature matrix $X = I$ and the adjacancy matrix $A$ of the graph structure.  The feature matrix $X$ is set as an identity matrix $I$, which simply means every word or document is represented as a one-hot vector.
Regarding the author, using the adjacancy matrix is efficient in situations when $X$ possess to less information. 


**Two layer GCN:**

First build a Adjacency and Degree

$$
\hat{\mathbf{A}} = \mathbf{D}^{1/2} \mathbf{A} \mathbf{D}^{1/2}
$$

and the layerwise linear model takes the form

$$
Z = f(X, A) = \text{softmax} \left( \hat{A} \text{ ReLu}(\hat{A}X W_0) W_1  \right)
$$  

where $W_0 \in \mathbb{R}^{C \times H}$ and $W_1 \in \mathbb{R}^{H \times F}$ are the network weights which are trained using gradient descent.
$X$ is a one-hot encoding of each of the graph nodes. 

For the training of a semi-supervised classifier the authors using full dataset batch gradient descent and evaluate the cross-entropy just error over the labeled data: 

$$
L = - \sum_{l \in \mathbb{y}_L} \sum_{f = 1}^{F} Y_{lf} ln Z_{lf}
$$,

with $\mathbb{y}_L$ as set of document idices that have labels and $F$ is the dimension of the output features, which is equal to the number of classes.


![]("Notebooks/GCN.png")

## Train 

For the trainig we need to tell the net work which nodes in the output layer (word - doc - embedding) is a labeled document node. As we have an semi supervised issue, some of the document embeddings have no label. The output matrix will have the shape of all nodes (words and documents) times the number of classes. Whereas the document indecies are the first. Hence, we just have to select the regarding output by the labeld node-indexes.
