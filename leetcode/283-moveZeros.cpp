/*** 
 * @Author: Lq
 * @Date: 2022-01-10 15:44:44
 * @LastEditTime: 2022-01-10 16:43:49
 * @LastEditors: Lq
 * @Description: 移动零
 */

/**
 * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
 * 
 * */

#include <vector>

using namespace std;

void swap(vector<int> &nums,int i,int j)
{
    int a = nums[i];
    nums[i] = nums[j];
    nums[j] = a;
}

/**
 * 双指针，left指向非0的子序列的尾部，
 * right 用来遍历数组，如果遇到非0的值，将其交换到i
 * */
void moveZeros(vector<int> &nums)
{
    int left = 0;
    int right = 0;
    while(right < nums.size()) {
        if(nums[right] != 0){
            swap(nums,left,right);
            left ++;
        }
        right ++;
    }
}