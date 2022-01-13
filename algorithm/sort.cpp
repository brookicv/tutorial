/*** 
 * @Author: Lq
 * @Date: 2022-01-11 14:13:19
 * @LastEditTime: 2022-01-12 11:06:25
 * @LastEditors: Lq
 * @Description: 经典的排序算法
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

// 插入排序，将待排序的数字，插入到前面已经排序的序列中
void insertSort(vector<int> &nums)
{
    for(size_t i = 1; i < nums.size(); i ++){
        auto val = nums[i]; // 当前待排序的值
        int j = i;
        // 将前面比val大的值向后移动，找到val的位置
        while(j > 0 && nums[j - 1] > val){
            nums[j] = nums[j-1];
            j --;
        }
        nums[j] = val;
    }
}

// 希尔排序，缩小增量排序，是插入排序的一种
void shellSort(vector<int> &nums)
{
    for(int gap = nums.size() / 2; gap > 0; gap /= 2){

        // 使用gap对数据进行分组，并逐个遍历
        for(int j = gap; j < nums.size(); j ++){
            int i = j;
            auto val = nums[j];
            // 在一个分组内使用插入排序
            // 遍历，移动比当前值大的数据，找到当前值的插入位置
            if(nums[i-gap] > nums[i]){
                while(i - gap >= 0 && nums[i - gap] > val){
                    nums[i] = nums[i - gap];
                    i -= gap;
                }
                nums[i] = val;
            }

        }
    }
}

// 冒泡排序，比较相邻两个位置的值，将较大的交换到后面
void bubbleSort(vector<int> &nums)
{
    for(int i = 0; i < nums.size(); i ++){
        bool swapFlag = false;
        for(int j = 0;j < nums.size() - 1; j ++){
            if(nums[j + 1] < nums[j]){
                int a = nums[j];
                nums[j] = nums[j + 1];
                nums[j + 1] = a;
                swapFlag = true;
            }
        }
        if(!swapFlag) break;
    }
}

// 每次遍历，将最小值的换到前面
void bubbleSort2(vector<int> &nums)
{
    for(int i = 0; i < nums.size();i ++){
        bool swapFlag = false;
        for(int j = i + 1; j < nums.size();j ++){
            if(nums[i] > nums[j]){
                int a = nums[i];
                nums[i] = nums[j];
                nums[j] = a;
                swapFlag = true;
            }
        }
        if(!swapFlag) break;
    }
}

// 每次选择最小（最大）的值放在当前位置
void selectSort(vector<int> &nums)
{
    for(int i = 0; i < nums.size() - 1; i ++){
        int min = nums[i];
        int minIdx = i;
        // 找到最小值
        for(int j = i + 1; j < nums.size(); j ++){
            if(nums[j] < min){
                min = nums[j];
                minIdx = j;
            }
        }

        // 将最小值和当前值进行交换
        if(i != minIdx){
            nums[minIdx] = nums[i];
            nums[i] = min;
        }
    }
}



int main()
{
    vector<int> nums;
    randomVector(nums,10);
    printVector(nums);
    shellSort(nums);
    printVector(nums);
}
