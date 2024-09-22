#include <bits/stdc++.h>

using namespace std; 

#define B 1;
#define R 0;

class node{
    public:
        node * p;
        node * l;
        node * r;
        int d;
        int c;
        node (int data) {
            p = NULL;
            l = NULL;
            r = NULL;
            d = data;
            c = R;
        }
};

class RBTree {

    public:
    node * root = NULL;

    void _search(node * temp, int x) {
        if (temp == NULL) {
            cout << "element not found" << endl;
            return;
        }
        if (temp->d == x) {
            cout << "element found" << endl;
            return;
        }

        else if (temp->d > x) {
            return _search(temp->l, x);
        }

        return _search(temp->r, x);
    }

    void search(int x) {
        node * temp = root;
        return _search(temp, x);
    }

    void insert(int e) {
        node * temp = root;
        node * new_node = new node(e);

        if (temp == NULL) { // root is null
            root = new_node;
            root->c = B;
        }

        while (temp != NULL) {
            if (temp->d > e) {
                if (temp->l == NULL) {
                    new_node->p = temp;
                    temp->l = new_node;
                    break;
                }
                temp = temp->l;
            }
            else {
                if (temp->r == NULL) {
                    new_node->p = temp;
                    temp->r = new_node;
                    break;
                }
                temp = temp->r;
            }
        }

    }

    void leftRotate(node * x) {
        node * y = x->r;
        node * yl = y->l;
        y->l = x;
        x->r = yl;
        
    }

    void _print_tree(node * temp) {
        if (temp != NULL) {
            cout << "(" << temp->d << " " << temp->c << ") ";
            _print_tree(temp->l);
            _print_tree(temp->r);
        }
    }

    void printTree() {
        node * temp = root;
        _print_tree(temp);
    }
};

int main () {
    RBTree * rbtree = new RBTree();
    rbtree->insert(34);
    rbtree->insert(12);
    rbtree->insert(13);
    rbtree->insert(16);
    rbtree->insert(32);
    rbtree->insert(19);
    rbtree->printTree();

    return 0;
}