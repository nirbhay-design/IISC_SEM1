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
        node (int data, int color = R) { // node constructor 
            p = NULL;
            l = NULL;
            r = NULL;
            d = data;
            c = color;
        }
};

class RBTree {

    public:
    node * nil = new node(INT_MIN, B); // initializing nil node
    node * root = nil; // root node 

    void _search(node * temp, int x) { // search function 
        if (temp == nil) {
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

    void _rb_insert_fixup(node * z) { // fixup code for red black tree insertion
        while (z->p != NULL && z->p->c == R) {
            if (z->p == z->p->p->l) {
                node * y = z->p->p->r;
                if (y->c == R) { // case 1
                    z->p->c = B;
                    z->p->p->c = R;
                    y->c = B;
                    z = z->p->p;
                } else {
                    if (z == z->p->r) { // case 2
                        z = z->p;
                        leftRotate(z);
                    }
                    z->p->c = B; // case 3
                    z->p->p->c = R;
                    rightRotate(z->p->p);
                }
            } else {
                node * y = z->p->p->l; // these are mirror image cases of prev ones
                if (y->c == R) {
                    z->p->c = B;
                    z->p->p->c = R;
                    y->c = B;
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

    void insert(int e) { // insertion code for Red Black Tree
        node * temp = root;
        node * new_node = new node(e);
        new_node->l = nil;
        new_node->r = nil;

        if (temp == nil) { // root is nil
            root = new_node;
            root->c = B;
            return;
        }

        while (temp != nil) { // finding the position to insert
            if (temp->d > e) {
                if (temp->l == nil) {
                    new_node->p = temp;
                    temp->l = new_node;
                    break;
                }
                temp = temp->l;
            }
            else {
                if (temp->r == nil) {
                    new_node->p = temp;
                    temp->r = new_node;
                    break;
                }
                temp = temp->r;
            }
        }

        if (new_node->p->c == R) { // fixup if node's parent color is Red
            _rb_insert_fixup(new_node);
        }
    }

    void leftRotate(node * x) { // left rotate 
        node * y = x->r;
        node * yl = y->l;

        y->l = x;
        x->r = yl;

        if (yl != nil) {
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

    void rightRotate(node * x) { // right rotate
        node * y = x->l;
        node * yr = y->r;

        y->r = x;
        x->l = yr;

        if (yr != nil) {
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

    void transplant(node * x, node * y) { // transplant function 
        if (x->p == NULL) { // checking parent is NULL 
            root = y;
            y->p = NULL;
            return;
        } 
        
        if (x == x->p->l) {
            x->p->l = y;
        } else {
            x->p->r = y;
        }

        y->p = x->p; // linking x->p to y
    }

    node * inorder_successor(node * x) { // finding the inorder successor of a node
        node * inos = x->r;
        while (inos->l != nil) {
            inos = inos->l;
        }
        return inos;
    }

    void cut(int e) { // deletion code 
        // need to write transplant node function -> Done
        // find inorder successor -> Done
        // detele fixup for red black tree -> Done
        if (root == nil) { // checking if root is nil
            cout << "Tree underflow" << endl;
            return;
        }

        node * temp = root;
        while (temp != nil && temp->d != e) {
            if (temp->d > e) {
                temp = temp->l;
            } else {
                temp = temp->r;
            }
        }

        if (temp == nil) { // checking if element exist
            cout << "element " << e << " does not exist in tree" << endl;
            return; 
        }

        node * x;
        int oc = temp->c; // color of cur node to be deleted;

        if (temp->l == nil) { // case 1
            x = temp->r;
            transplant(temp, temp->r);
        } else if (temp->r == nil) { // case 2
            x = temp->l;
            transplant(temp, temp->l);
        } else { // case 3
            node * inos = inorder_successor(temp);
            oc = inos->c;
            x = inos->r;
            if (inos != temp->r) {
                transplant(inos, inos->r);
                inos->r = temp->r;
                inos->r->p = temp;
            } else {
                x->p = inos;
            }
            transplant(temp, inos);
            inos->l = temp->l;
            inos->l->p = inos;
            inos->c = temp->c;
        }
        // printTree();
        // cout << "calling delete fixup" << endl;
        if (oc == B){ // calling fixup if color was Black
            _rb_delete_fixup(x);
        }

    }

    void _rb_delete_fixup(node * x) {
        while (x != root && x->c == B) {
            if (x == x->p->l) {
                node * w  = x->p->r; // sibling
                if (w->c == R) { // case 1
                    w->c = B;
                    x->p->c = R;
                    leftRotate(x->p);
                    w = x->p->r;
                }
                if (w->l->c == B && w->r->c == B) { // case 2
                    w->c = R;
                    x = x->p;
                } else {
                    if (w->r->c == B) { // case 3
                        w->l->c = B;
                        w->c = R;
                        rightRotate(w);
                        w = x->p->r;
                    }
                    w->c = x->p->c; // case 4
                    x->p->c = B;
                    w->r->c = B;
                    leftRotate(x->p);
                    x=root;
                }
            } else {
                node * w  = x->p->l; // mirror image of above cases
                if (w->c == R) {
                    w->c = B;
                    x->p->c = R;
                    rightRotate(x->p);
                    w = x->p->l;
                }
                if (w->l->c == B && w->r->c == B) {
                    w->c = R;
                    x = x->p;
                } else {
                    if (w->l->c == B) {
                        w->r->c = B;
                        w->c = R;
                        leftRotate(w);
                        w = x->p->l;
                    }
                    w->c = x->p->c;
                    x->p->c = B;
                    w->l->c = B;
                    rightRotate(x->p);
                    x=root;
                }
            }
        }
        x->c = B;
    }

    void _print_tree(node * temp) { // printing preorder of the tree
        if (temp != NULL) {
            // if (temp->d == INT_MIN) cout << " NIL ";
            if (temp->d != INT_MIN) cout << "(" << temp->d << " " << (temp->c ? "B":"R") << ") ";
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

    while (true){
        cout << "Delete [0] / Insert [1]: ";
        int decide, e; cin >> decide;
        if (decide) {
            cout << "enter element to insert: ";
            cin >> e;
            rbtree->insert(e);
            rbtree->printTree();
        } else {
            cout << "enter element to delete: ";
            cin >> e;
            rbtree->cut(e);
            rbtree->printTree();
        }
        
    }
    // rbtree->insert(34);
    // rbtree->printTree();
    // rbtree->insert(48);
    // rbtree->printTree();
    // rbtree->insert(43);
    // rbtree->printTree();
    // rbtree->insert(52);
    // rbtree->printTree();
    // rbtree->insert(32);
    // rbtree->printTree();
    // rbtree->insert(19);
    // rbtree->printTree();
    // rbtree->insert(33);
    // rbtree->printTree();
    // rbtree->insert(12);
    // rbtree->printTree();

    // node * is = rbtree->inorder_successor(rbtree->root->r);
    // cout << is->d << endl;
    return 0;
}