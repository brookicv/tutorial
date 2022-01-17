/*** 
 * @Author: Lq
 * @Date: 2022-01-17 15:23:56
 * @LastEditTime: 2022-01-17 17:42:16
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
}


int main()
{
    string str = "hello";
    string p = "aa";

    auto it = str.find_first_of("",0);
    cout << it << endl;

    cout << strStr(str,p) << endl;
    return 0;
}
