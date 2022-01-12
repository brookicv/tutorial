/*** 
 * @Author: Lq
 * @Date: 2022-01-12 18:26:07
 * @LastEditTime: 2022-01-12 18:39:19
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

ListNode* paritalLink(ListNode* head){
    auto node = head;
    int length = 0;
    while(node){
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


int main()
{
    ListNode* head = generate(11);
    printLink(head);

    ListNode* right = paritalLink(head);
    printLink(right);

    return 0;
}