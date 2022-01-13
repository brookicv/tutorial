/*** 
 * @Author: Lq
 * @Date: 2022-01-12 18:26:07
 * @LastEditTime: 2022-01-13 11:00:56
 * @LastEditors: Lq
 * @Description: 单链表排序
 */

#include <iostream>
#include <random>

using namespace std;

struct ListNode {
     int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
 };

ListNode* generate(int size)
{
    static random_device rd; // 产生随机数种子
    static mt19937 mt(rd());
    static uniform_int_distribution<int> gen(0,size);

    ListNode* head = new ListNode(gen(mt),nullptr);
    ListNode* currtNode = head;
    for(int i = 1; i < size; i ++){
        auto node = new ListNode(gen(mt));
        node->next = nullptr;
        currtNode->next = node;
        currtNode = node;
    }

    return head;
}

void printLink(ListNode* head)
{
    auto node = head;
    while(node){
        cout << node->val << " ";
        node = node->next;
    }
    cout << endl;
}

void destoryLink(ListNode* head){
    auto node = head;
    while(node){
        auto next = node->next;
        free(node);
        node = next;
    }
}

ListNode* paritalLink(ListNode* head,ListNode* end){
    auto node = head;
    int length = 0;
    while(node != end){
       length ++;
       node = node->next;
    }

    int index = 0;
    node = head;
    while(index < length / 2){
        index ++;
        node = node->next;
    }

    return node;
}

ListNode* findMid(ListNode* head)
{
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    auto fast = head->next;
    auto slow = head;
    while(fast != nullptr && fast->next != nullptr){
        slow = slow->next;
        fast = fast->next->next;
    }

    return slow;
}

ListNode* merge(ListNode *head1,ListNode *head2)
{
    if(head1 == nullptr && head2 == nullptr){
        return nullptr;
    }

    ListNode *head = nullptr;
    ListNode *current = nullptr;
    while(head1 && head2){
        if(head1->val > head2->val){
            if(head == nullptr){
                head = head2;
                current = head;
            }else {
                current->next = head2;
                current = current->next;
            }
            head2 = head2->next;
        } else {
            if(head == nullptr){
                head = head1;
                current = head;
            } else {
                current->next = head1;
                current = current->next;
            }
            head1 = head1->next;
        }
    }

    if(head1 != nullptr){
        current->next = head1;
    }

    if(head2 != nullptr){
        current->next = head2;
    }

    return head;
}



ListNode* mergeSort(ListNode *head)
{
    if(head == nullptr || head->next == nullptr){
        return head;
    }

    auto mid = findMid(head);
    auto right = mergeSort(mid->next);

    mid->next = nullptr;
    auto left = mergeSort(head);
    return merge(left,right);
}


int main()
{
    ListNode* head = generate(11);
    printLink(head);

    ListNode* right = paritalLink(head,nullptr);
    printLink(right);

    ListNode* mid = findMid(head);
    printLink(mid);

    ListNode* sort = mergeSort(head);
    printLink(sort);

    return 0;
}