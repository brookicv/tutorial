/*** 
 * @Author: Lq
 * @Date: 2022-01-14 09:36:04
 * @LastEditTime: 2022-01-14 16:06:42
 * @LastEditors: Lq
 * @Description: 摆动排序，锯齿排序
 */

#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

void printVector(vector<int> &nums)
{
    for(auto i: nums){
        cout << i << " ";
    }
    cout << endl;
}

void swap(vector<int> &nums,int i,int j)
{
    int a = nums[i];
    nums[i] = nums[j];
    nums[j] = a;
}
void wiggleSort(vector<int> &nums)
{
    auto helper = nums;
    int n = helper.size();
    sort(helper.begin(),helper.end());

    for(int i = 1; i < helper.size(); i += 2){
        nums[i] = helper[--n];
    }

    for(int i = 0; i < helper.size(); i += 2){
        nums[i] = helper[--n];
    }
}


void wiggleSort2(vector<int> &nums)
{
    auto helpr = nums;
    int n = helpr.size();

    auto midIt = helpr.begin() + helpr.size() / 2;
    // 左侧元素不大于mid，右侧元素不小于自身。
    // 如果mid元素有重复，则可能出现在左边，也有可能出现在右边
    nth_element(helpr.begin(),midIt,helpr.end());

    int mid = *midIt;

    // 3-way-parition
    int i = 0,j = 0;
    int k = helpr.size() - 1;
    while(j < k){
        if(helpr[j] > mid){
            swap(helpr,j,k); // 将 nums[k]交换到j处，继续比较j处的值和mid的大小
            k --;
        } else if(helpr[j] < mid){
            swap(helpr,j,i); // 将nums[i]交换到j处，j本身是从左边开始的，i在mid的前面，一定是小于等于mid的
            ++i; 
            ++j;
        } else {
            ++j;
        }
    }

    for(int i = 1; i < helpr.size(); i +=2){
        nums[i] = helpr[--n];
    }

    for(int i = 0; i < helpr.size(); i +=2){
        nums[i] = helpr[--n];
    }
}

void quickSelect(vector<int> &nums,int begin,int end,int nth)
{
    int val = nums[end - 1]; // 选择最后一个值作为基准点
    int i = begin; // 指向数组左边的下标，小于基准值的放在左边
    int j = begin; // 遍历数组的下标
    while(j < end){
        if(nums[j] <= val){
            swap(nums,j,i);
            j ++;
            i ++;
        }else {
            j ++;
        }
    }

    if( i - 1 > nth){
        quickSelect(nums,begin,i - 1,nth);
    }else if (i <= nth){
        quickSelect(nums,i,end,nth);
    }
}

int main()
{
    vector<int> nums = {1,3,2,2,3,2};
    auto a = nums;
    auto midItator = a.begin() + nums.size() / 2;
    nth_element(a.begin(),midItator,a.end());

    printVector(a);
    quickSelect(nums,0,nums.size(),nums.size() / 2);

    printVector(nums);
}
