#include <bits/stdc++.h>

using namespace std; 

#define B 1
#define R 0

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

    void _rb_insert_fixup(node * z) {
        while (z != NULL && z->p != NULL && z->p->c == R) {
            if (z->p == z->p->p->l) {
                node * y = z->p->p->r;
                if (y != NULL && y->c == R) {
                    z->p->c = B;
                    z->p->p->c = R;
                    if (y != NULL) y->c = B;
                    z = z->p->p;
                } else {
                    if (z == z->p->r) {
                        z = z->p;
                        leftRotate(z);
                    }
                    z->p->c = B;
                    z->p->p->c = R;
                    rightRotate(z->p->p);
                }
            } else {
                node * y = z->p->p->l;
                if (y != NULL && y->c == R) {
                    z->p->c = B;
                    z->p->p->c = R;
                    if (y != NULL) y->c = B;
                    z = z->p->p;
                } else {
                    if (z == z->p->l) {
                        z = z->p;
                        rightRotate(z);
                    }
                    z->p->c = B;
                    z->p->p->c = R;
                    leftRotate(z->p->p);
                }
            }   
        }
        root->c = B;
    }

    void insert(int e) {
        node * temp = root;
        node * new_node = new node(e);

        if (temp == NULL) { // root is null
            root = new_node;
            root->c = B;
            return;
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
        if (new_node->p->c == R) { 
            _rb_insert_fixup(new_node);
        }
    }

    void leftRotate(node * x) {
        node * y = x->r;
        node * yl = y->l;

        y->l = x;
        x->r = yl;

        if (yl != NULL) {
            yl->p = x;
        }

        if (x == root) {
            root = y;
        } else {
            if (x == x->p->l) {
                x->p->l = y;
            } else {
                x->p->r = y;
            }
        }
        
        y->p = x->p;
        x->p = y;
    }

    void rightRotate(node * x) {
        node * y = x->l;
        node * yr = y->r;

        y->r = x;
        x->l = yr;

        if (yr != NULL) {
            yr->p = x;
        }

        if (x == root) {
            root = y;
        } else {
            if (x == x->p->l) {
                x->p->l = y;
            } else {
                x->p->r = y;
            }
        }

        y->p = x->p;
        x->p = y;

    }

    void cut(int e) {
        // deletion code to be written here;
        // need to write transplant node function
        // find inorder successor
        // detele fixup for red black tree
    }

    void _print_tree(node * temp) {
        if (temp != NULL) {
            cout << "(" << temp->d << " " << (temp->c ? "B":"R") << ") ";
            _print_tree(temp->l);
            _print_tree(temp->r);
        }
    }

    void printTree() {
        node * temp = root;
        _print_tree(temp);
        cout << endl;
    }
};

int main () {
    RBTree * rbtree = new RBTree();
    rbtree->insert(34);
    rbtree->printTree();
    rbtree->insert(48);
    rbtree->printTree();
    rbtree->insert(43);
    rbtree->printTree();
    rbtree->insert(52);
    rbtree->printTree();
    rbtree->insert(32);
    rbtree->printTree();
    rbtree->insert(19);
    rbtree->printTree();
    rbtree->insert(33);
    rbtree->printTree();
    rbtree->insert(12);
    rbtree->printTree();

    return 0;
}