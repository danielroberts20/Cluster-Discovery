import io
import os
import zipfile
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import requests
from nltk import download
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold


class Comp3222:

    co_matrix = None
    dataset = None
    silhouettes = []
    inertias = []

    def __init__(self,
                 min_k:int = 3,
                 max_k:int = 15,
                 short_word_length:int = 2,
                 word_threshold:int = 25,
                 folds:int = 5,
                 verbose:bool = False):

        self.min_k = min_k
        self.max_k = max_k
        self.short_word_length = short_word_length
        self.word_threshold = word_threshold
        self.folds = folds
        self.verbose = verbose

    def set_verbose(self, verbose:bool):
        self.verbose = verbose

    def load_matrix(self, path):
        if os.path.exists(path):
            self._log(f"Loading co-occurrence matrix from {path}...")
            self.co_matrix = np.load(path)

    def load_dataset(self, path):
        with open(path, "r") as f:
            self._log(f"Loading dataset from {path}...")
            self.dataset = f.read().split()

    def save_matrix(self, path):
        np.save(path, self.co_matrix)

    def save_dataset(self, path):
        with open(path, 'w+') as out:
            out.write(" ".join(self.dataset))


    def _log(self, msg):
        """
        Used for logging. Displays message if verbose is on.
        :param msg:
        :return:
        """
        if self.verbose:
            print(msg)


    def _create_cooccurrence_matrix(self, words, window_size: int = 7):
        """
        Create the co-occurrence matrix.
        :param words: The (in order) dataset.
        :param window_size: The number of words either side of the current word to check.
                            (optional, default: 7)
        :return: The co-occurrence matrix of shape (vocab_size, vocab_size).
        """
        if os.path.exists("co-matrix.npy"):
            self._log("Loading co-occurrence matrix from file...")
            return np.load("co-matrix.npy")
        else:
            # Get unique words in the input list
            self._log("Creating co-occurrence matrix...")
            voc = set(words)
            self._log(f"\nCreating vocab (size={len(voc)})...")

            word_to_index = {word: idx for idx, word in enumerate(voc)}

            # Initialize a (sparse) square matrix with zeros
            cooccurrence_matrix = lil_matrix((len(voc), len(voc)))

            self._log(f"Populating co-occurrence matrix...")

            # The following code to populate a co-occurrence matrix was taken from the MLT lab 3.
            for i, word in enumerate(words):
                word_idx = word_to_index[word]
                # Look at the surrounding words within the window size
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(words))

                # Start on the left side of the window, work up to the right side.
                for j in range(start, end):
                    if i != j:  # Avoid self co-occurrence
                        neighbor_word = words[j]
                        neighbor_idx = word_to_index[neighbor_word]
                        cooccurrence_matrix[word_idx, neighbor_idx] += 1
            self._log(f"Populated co-occurrence matrix. Shape: {cooccurrence_matrix.shape}")
            return cooccurrence_matrix


    def _pre_process(self, path):
        """
        Pre-process the dataset. Default values result in a 41.89% reduction in size
        from 17005207 words to 9882180.
        :param path: Path to dataset file. (optional, default: "text8")
        :return: List of strings (words).
        """

        # If the dataset has not already been processed, but has been downloaded
        try:
            with open(path, "r") as file:
                self._log("Pre-processing data...")
                # If the dataset is stored locally
                text = file.read().split()

                # Remove all words less than 2 characters or longer than 18 characters
                self._log(f"Removing words less than "
                          f"{self.short_word_length} characters...")
                short_long = [w for w in text if self.short_word_length < len(w)]
                self._log(f"\t| {len(text)} words -> {len(short_long)} words "
                          f"({((len(short_long) - len(text)) / len(text)) * 100:.2f}%)")

                # Remove all stopwords
                self._log("Removing stopwords...")
                download("stopwords", quiet=True)
                stops = set(stopwords.words("english"))
                stopped = [w for w in short_long if w not in stops]
                self._log(f"\t| {len(short_long)} words -> {len(stopped)} words "
                          f"({((len(stopped) - len(short_long)) / len(short_long)) * 100:.2f}%)")

                # Remove words that appear less than X times
                self._log(f"Removing words that appear less than {self.word_threshold} times...")
                freq = Counter(stopped)
                thresh_words = set([w for w, c in freq.items() if c < self.word_threshold])
                thresholded_words = [w for w in stopped if w not in thresh_words]
                self._log(f"\t| {len(stopped)} words -> {len(thresholded_words)} words "
                          f"({((len(thresholded_words) - len(stopped)) / len(stopped)) * 100:.2f}%)")

                self._log("Finished pre-processing.")
                self._log(f"\t| {len(text)} words -> {len(thresholded_words)} words "
                          f"({((len(thresholded_words) - len(text)) / len(text)) * 100:.2f}%)")

                return thresholded_words
        except FileNotFoundError:
            # If the dataset is not stored locally, download and extract it, then process it
            self._log("Dataset not found locally. Downloading...")
            response = requests.get("https://mattmahoney.net/dc/text8.zip")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall()
                os.rename(zip_ref.namelist()[0], path)
            return self._pre_process(path)

    def fit(self, path="text8"):
        if self.dataset is None:
            self.dataset = self._pre_process(path)
        if self.co_matrix is None:
            self.co_matrix = self._create_cooccurrence_matrix(self.dataset)


    def predict(self):
        """
        Search for the ideal k value.

        :return: The range of k values searched. The average silhouette score for each k value. The average inertia for each k value.
        """

        # Create k-fold cross-validation
        kf = KFold(n_splits=self.folds)

        scores = []  # Holds average silhouette scores for each k
        elbow = []  # Holds average inertia for each k
        ks = range(self.min_k, self.max_k + 1)
        for k in ks:
            avg_score = 0
            avg_inertia = 0
            self._log(f"Training KMeans model (k={k})...")
            for i, (train_index, test_index) in enumerate(kf.split(self.co_matrix)):  # Each fold
                self._log(f"Fold {i + 1}/{self.folds}")

                # Convert to sparse matrices. Improves speed.
                X_train, X_test = self.co_matrix[train_index], self.co_matrix[test_index]

                # Random state corresponds to the current fold.
                # This means that the i-th fold of every k value will be the same.
                kmeans = KMeans(n_clusters=k, random_state=i)
                kmeans.fit(X_train)  # Learn from training data
                predictions = kmeans.predict(X_test)  # Evaluate on validation data
                avg_score += silhouette_score(X_test, predictions)
                avg_inertia += kmeans.inertia_
            scores.append(avg_score / self.folds)  # Average silhouette score
            elbow.append(avg_inertia / self.folds)  # Average inertia
        self.silhouettes = scores
        self.inertias = elbow

    def plot(self) -> None:
        """
        Plot the silhouette scores and inertias for all k values searched.
        Allows human to judge best k visually.
        """
        # Plot the inertias (elbow method)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.min_k, self.max_k + 1), self.inertias, marker='o')
        plt.title(f"Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid(True)

        # Plot the silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(range(self.min_k, self.max_k + 1), self.silhouettes, marker='o')
        plt.title(f"Silhouette Score")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)

        # Display
        plt.tight_layout()
        plt.show()


def main():

    model = Comp3222()
    model.set_verbose(True)
    model.fit()
    model.predict()
    model.plot()

if __name__ == "__main__":
    main()
