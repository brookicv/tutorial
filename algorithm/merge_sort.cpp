/*** 
 * @Author: Lq
 * @Date: 2022-01-12 17:01:18
 * @LastEditTime: 2022-01-12 18:25:02
 * @LastEditors: Lq
 * @Description: 归并排序
 */

/*
    归并排序，利用分治思想，将要排序的数组分割为两个部分分别排序，再合并到一起
*/

#include <vector>
#include <random>
#include <iostream>

using namespace std;

void randomVector(vector<int> &a,int size=20)
{
    static random_device rd; // 产生随机数种子
    static mt19937 mt(rd());
    static uniform_int_distribution<int> gen(0,size);

    for(int i = 0; i < size; i ++){
        a.push_back(gen(mt));
    }
}

void printVector(const vector<int> &a)
{
    for(const auto &i:a){
        cout << i << " ";
    }
    cout << endl;
}



void merge(vector<int> &nums,int left,int mid,int right,vector<int> &tmp)
{
    int i = left;
    int j = mid + 1;
    // tmp index
    int idx = 0;
    while( i <= mid && j <= right){
        if(nums[i] < nums[j]){
            tmp[idx++] = nums[i++];
        }else {
            tmp[idx++] = nums[j++];
        }
    }

    while(i <= mid){
        tmp[idx++] = nums[i++];
    }

    while(j <= right){
        tmp[idx++] = nums[j++];
    }

    idx = 0;
    // 将合并后的数组复制到源数组中
    while(left <= right){
        nums[left++] = tmp[idx++];
    }
}



void paritalSort(vector<int> &nums,int left,int right,vector<int> &tmp)
{
    if(left < right){
        int mid = (left + right) / 2;
        paritalSort(nums,left,mid,tmp);
        paritalSort(nums,mid + 1,right,tmp);

        // 合并
        merge(nums,left,mid,right,tmp);
    }
}

void paritalSortLoop(vector<int> &nums)
{
    int left = 0;
    int right = nums.size()- 1;

    
}

void mergeSort(vector<int> &nums)
{
    vector<int> tmp(nums.size());
    paritalSort(nums,0,nums.size() - 1,tmp);
}

int main()
{
    vector<int> nums;
    randomVector(nums,5);
    printVector(nums);
    mergeSort(nums);
    printVector(nums);
    return 0;
}
