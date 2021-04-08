import math

def unique_vals(rows, col):
    #Find the unique values for a column in a dataset.
    return set([row[col] for row in rows])

"""
Input:-
#rows: list of Dataset
#col: column number

Output:- list of unique values in the column
"""


def class_counts(rows):
    #Counts the number of each type of example in a dataset.
    counts = {}  # a dictionary of: label -> count.
    for row in rows:
        # in our dataset format, the label is the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

"""
Input:-
#rows: list of Dataset
Output:- a dictionary of: label -> count 
"""

# a helper function
def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label

# a helper function
def is_numeric(value):
    #Test if a value is numeric.
    return isinstance(value, int) or isinstance(value, float)



class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 4 for 'Booking_Type') and a
    'column value' (e.g., 'Group bookings'). 
    The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question.
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, checks if it satisfies the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculates the Gini Impurity for a list of rows.
    Gini impurity is a measure of how often a randomly chosen element from the set 
    would be incorrectly labeled if it was randomly labeled according to the distribution 
    of labels in the subset.
    There are a few different ways to do this, 
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def entropy(rows):

    # compute the entropy.
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(len(rows))
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy


def info_gain(left, right, current_uncertainty, algo):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))

    if algo == 0:
        return current_uncertainty - p * gini(left) - (1-p) * gini(right)
    if algo == 1:
        return current_uncertainty - p * entropy(left) - (1-p) * entropy(right)

def find_best_split(rows, header, algo):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    if algo == 0:
        current_uncertainty = gini(rows)
    if algo == 1:
        current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val, header)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty, algo)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows


def build_tree(rows, header, algo, depth=0, id=0):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    # depth = 0
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    gain, question = find_best_split(rows, header, algo)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, id, depth)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    # nodeLst.append(id)
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header, algo, depth + 1, 2 * id + 2)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header, algo, depth + 1, 2 * id + 1)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on on the answer.
    return Decision_Node(question, true_branch, false_branch, depth, id, rows)


def prune_tree(node, prunedList):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node
    # If we reach a pruned node, make that node a leaf node and return. Since it becomes a leaf node, the nodes
    # below it are automatically not considered
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    # Call this function recursively on the true branch
    node.true_branch = prune_tree(node.true_branch, prunedList)

    # Call this function recursively on the false branch
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


def classify(row, node):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predicted_label

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


#tree printing function
def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions) + " Label Class: " + str(node.predicted_label))
        return

    # Print the question at this node
    print(spacing + str(node.question) + " id: " + str(node.id) + " depth: " + str(node.depth))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def getLeafNodes(node, leafNodes =[]):

    # Base case
    if isinstance(node, Leaf):
        leafNodes.append(node)
        return

    # Recursive right call for true values
    getLeafNodes(node.true_branch, leafNodes)

    # Recursive left call for false values
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes =[]):

    # Base case
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    # Recursive right call for true values
    getInnerNodes(node.true_branch, innerNodes)

    # Recursive left call for false values
    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes


def computeAccuracy(rows, node):

    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        # last entry of the column is the actual label
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy/count, 5)