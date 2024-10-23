#include <iostream>
using namespace std;

class RBTreeNode
{
public:
    string color;
    int key;
    RBTreeNode *p, *left, *right;
    // Constructor to initialize node with value
    RBTreeNode(int val)
    {
        key = val;
        p = left = right = NULL;
        color = "red"; // Default new nodes as red in Red-Black Tree
    }
    // Default constructor
    RBTreeNode()
    {
        key = 0;
        p = left = right = NULL;
        color = "black"; // Default as black for nil nodes
    }
};

class RBTree
{
public:
    RBTreeNode *root, *nil;

    // Constructor for Red-Black Tree
    RBTree()
    {
        // Initialize the nil node
        nil = new RBTreeNode();
        nil->color = "black";                  // Nil node is always black
        nil->left = nil->right = nil->p = nil; // Nil's children and parent point to nil

        // Initialize the root as nil
        root = nil;
    }

    // Left Rotate
    void leftRotate(RBTreeNode *x)
    {
        RBTreeNode *y = x->right; // Set y
        x->right = y->left;       // Turn y's left subtree into x's right subtree
        if (y->left != nil)
            y->left->p = x;

        y->p = x->p; // Link x's parent to y
        if (x->p == nil)
            root = y;
        else if (x == x->p->left)
            x->p->left = y;
        else
            x->p->right = y;

        y->left = x; // Put x on y's left
        x->p = y;
    }

    // Right Rotate
    void rightRotate(RBTreeNode *y)
    {
        RBTreeNode *x = y->left; // Set x
        y->left = x->right;      // Turn x's right subtree into y's left subtree
        if (x->right != nil)
            x->right->p = y;

        x->p = y->p; // Link y's parent to x
        if (y->p == nil)
            root = x;
        else if (y == y->p->right)
            y->p->right = x;
        else
            y->p->left = x;

        x->right = y; // Put y on x's right
        y->p = x;
    }

    // Fix Red-Black Tree after insertion
    void RB_insert_fixup(RBTreeNode *node)
    {
        while (node->p->color == "red")
        {
            if (node->p == node->p->p->left)
            {
                RBTreeNode *uncle = node->p->p->right;

                // Case 1: Uncle is red (recoloring)
                if (uncle->color == "red")
                {
                    node->p->color = "black";
                    uncle->color = "black";
                    node->p->p->color = "red";
                    node = node->p->p;
                }
                else
                {
                    // Case 2: Node is right child (left rotation)
                    if (node == node->p->right)
                    {
                        node = node->p;
                        leftRotate(node);
                    }
                    // Case 3: Node is left child (right rotation)
                    node->p->color = "black";
                    node->p->p->color = "red";
                    rightRotate(node->p->p);
                }
            }
            else
            { // Mirror cases for the right child
                RBTreeNode *uncle = node->p->p->left;

                // Case 1: Uncle is red (recoloring)
                if (uncle->color == "red")
                {
                    node->p->color = "black";
                    uncle->color = "black";
                    node->p->p->color = "red";
                    node = node->p->p;
                }
                else
                {
                    // Case 2: Node is left child (right rotation)
                    if (node == node->p->left)
                    {
                        node = node->p;
                        rightRotate(node);
                    }
                    // Case 3: Node is right child (left rotation)
                    node->p->color = "black";
                    node->p->p->color = "red";
                    leftRotate(node->p->p);
                }
            }
        }
        root->color = "black"; // Root is always black
    }

    // Insert function
    void insert(int val)
    {
        cout << "Inserting: " << val << "\n";
        RBTreeNode *z = new RBTreeNode(val);
        RBTreeNode *x = root;
        RBTreeNode *y = nil;

        while (x != nil)
        {
            y = x;
            if (z->key < x->key)
                x = x->left;
            else
                x = x->right;
        }

        z->p = y;
        if (y == nil)
            root = z; // Tree was empty
        else if (z->key < y->key)
            y->left = z;
        else
            y->right = z;

        z->left = z->right = nil;
        z->color = "red";

        RB_insert_fixup(z);
    }

    // Depth-first search (DFS) to print the tree (in-order traversal)
    void dfs(RBTreeNode *node)
    {
        if (node == nil)
        {
            return;
        }
        cout << "(" << node->key << ", " << node->color << ") ";
        dfs(node->left);
        dfs(node->right);
    }

    // Print the tree
    void printTree()
    {
        dfs(root);
        cout << endl;
    }
};

int main()
{
    // Initialize Red-Black Tree
    RBTree *T = new RBTree();

    // Input the number of nodes
    cout << "enter number of nodes: ";
    int n;
    cin >> n;

    // Insert nodes
    for (int i = 0; i < n; ++i)
    {
        int val;
        cout << "enter value to insert: ";
        cin >> val;
        T->insert(val);
        T->printTree();
        cout << endl;
    }

    // Print the Red-Black Tree (In-order traversal)

    return 0;
}