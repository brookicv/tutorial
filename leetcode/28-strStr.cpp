/*** 
 * @Author: Lq
 * @Date: 2022-01-17 15:23:56
 * @LastEditTime: 2022-01-18 16:45:05
 * @LastEditors: Lq
 * @Description: 字符串匹配，在目标字符串中找出模式串的最开始出现的问题
 */

#include <string>
#include <vector>
#include <iostream>

using namespace std;

int strStr(const string &str,const string &p,int pos = 0)
{
    if(p.empty()) return 0;

    int i = pos;
    int j = 0;

    while(i < str.size() && j < p.size()){
        if(str[i] == p[j]){
            i ++;
            j ++;
        }else {
            i = i - j + 1; // 回溯到第一个不匹配的地方
            j = 0;
        }
    }

    if(j == p.size()){
        return i - j;
    }else {
        return -1;
    }
}

void printVector(const vector<int> &nums)
{
    for(int a : nums){
        cout << a << " ";
    }
    cout << endl;
}

int kmp(const string &str,const string &p,int pos = 0)
{
    if(p.empty()) return 0;

    vector<int> next(p.length());

    next[0] = -1;

    int j = 0;
    int k = -1;

    while(j < p.length() - 1){
        if(k == -1 || p[k] == p[j]){
            k ++;
            j ++;
            next[j] = next[k];
        }else {
            k = next[k];
        }
    }

    printVector(next);

}


int kmp2(const string &str,const string &pattern)
{
    vector<int> next(pattern.size());

    next[0] = 0;
    int left = 0; // 即使指向前缀的指针，也是其长度
    int right = 1;
    while (right < pattern.size()){

        while(left > 0 && pattern[left] != pattern[right]){
            left = next[left - 1]; // 回溯,直到回退到0，从头开始
        }
        if(pattern[left] == pattern[right]){
            left ++; // 继续，+1
        }
        next[right] = left;
        right ++;
    }

    printVector(next);

    int i = 0; 
    int j = 0;
    while(i < str.size() && j < pattern.size()){

        while(j > 0 && str[i] != pattern[j]){
            j = next[j - 1];
        }
        if(str[i] == pattern[j]){
            j ++;
        }

        if(j == pattern.size()){
            return i - j + 1;
        }

        i ++;
    }

    return -1;
}


int main()
{
    string str = "hello";
    string p = "ll";

    cout << kmp2(str,p) << endl;
    return 0;
}
