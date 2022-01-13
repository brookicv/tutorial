/*** 
 * @Author: Lq
 * @Date: 2022-01-12 14:53:21
 * @LastEditTime: 2022-01-12 15:59:20
 * @LastEditors: Lq
 * @Description: 堆排序
 */


/*
    堆，使用数组存储的数据结构，是一棵完全二叉树。
    节点i的，左子节点为 2 * i + 1; 右子节点为 2 * i + 2;
    最大堆，nums[i] >= nums[2 * i + 1] && nums[i] >= nums[2 * i + 2]
    最小堆，nums[i] <= nums[2 * i + 1] && nums[i] <= nums[2 * i + 2]
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

void maxHeapAdjust(vector<int> &nums,int start,int end)
{
    auto val = nums[start]; // 取出当前元素

    // 从节点start的左右子树中找出最大的，放在节点start处
    for(int i = 2 * start + 1; i < end; i = 2 * i+ 1){

        // 左右节点最大的
        if(i + 1 < end && nums[i + 1] > nums[i]){
            i ++;
        }
        if(nums[i] > val){
            nums[start] = nums[i];
            start = i;
        }else {
            break;
        }
    }
    nums[start] = val;
}

void maxHeapSort(vector<int> &nums)
{
    // 构建最大堆，从第一个非叶子节点开始，从下往上，从左到右
    for(int i = nums.size() / 2 - 1; i >= 0; i --){
        maxHeapAdjust(nums,i,nums.size());
    }

    for(int i = nums.size() - 1; i > 0; i--){
        // 取出堆的元素，为当前最大的元素，放在数列尾部，升序排序
        auto val = nums[i];
        nums[i] = nums[0];
        nums[0] = val;
        // 重新调整堆
        maxHeapAdjust(nums,0,i);
    }
}

void smallHeapAdjust(vector<int> &nums,int start,int end)
{
    auto val = nums[start];

    // 调整
    // 在左右子节点中找到最小的和当前节点的值进行交换
    for(int i = 2 * start + 1; i < end; i = 2 * i + 1){
        if( i + 1 < end && nums[i + 1] < nums[i]){
            i ++ ;
        }
        if(nums[i] < val){
            nums[start] = nums[i];
            start = i;
        }else {
            break;
        }
    }
    nums[start] = val;
}

void smallHeapSort(vector<int> &nums)
{
    // 构建最小堆，从最后一个非叶子节点开始
    for(int i = nums.size() / 2 - 1; i >= 0; i --){
        smallHeapAdjust(nums,0,nums.size());
    }

    for(int i = nums.size() - 1; i > 0; i --){
        auto val = nums[i];
        nums[i] = nums[0];
        nums[0] = val;
        smallHeapAdjust(nums,0,i);
    }
}

int main()
{
    vector<int> nums;
    randomVector(nums,5);
    printVector(nums);
    smallHeapSort(nums);
    printVector(nums);
    return 0;
}


