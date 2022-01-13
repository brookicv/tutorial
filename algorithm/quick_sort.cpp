/*** 
 * @Author: Lq
 * @Date: 2022-01-13 14:30:46
 * @LastEditTime: 2022-01-13 16:54:43
 * @LastEditors: Lq
 * @Description: 快速排序
 */

/*
    通过一趟排序将要排序的数据分割成独立的两部分，
    其中一部分的所有数据都比另外一部分的所有数据都要小，
    然后再按此方法对这两部分数据分别进行快速排序，
    整个排序过程可以递归进行，以此达到整个数据变成有序序列。
*/

#include <vector>
#include <random>
#include <iostream>
#include <ctime>
#include <cstdlib>

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

int randomIndex(int min,int max)
{
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> gen(min,max);

    return gen(mt);
}

void swap(vector<int> &nums,int i,int j)
{
    int a = nums[i];
    nums[i] = nums[j];
    nums[j] = a;
}

int randomPartition(vector<int> &nums,int left,int right)
{
    // 随机选择一个下标作为基准点
    int index = randomIndex(left,right);
    // int index = rand() % (right - left + 1) +left;

    // 将随机选择的基准数据放到left位置
    swap(nums,index,left);

    int val = nums[left];
    
    index = left;
    for(int i = left + 1; i <= right; i ++){

        // 如果小于基准数据就放到数组的前面，
        // 最后index前面的数据都小于基准数据，
        // 将基准数据交换到index位置
        if(nums[i] < val){
            index ++;
            if(index != i) {
                swap(nums,index,i);
            }
            
        }
    }
    swap(nums,index,left);
    return index;
}

void randomQuickSort(vector<int> &nums,int left,int right)
{
    if(left < right){
        auto p = randomPartition(nums,left,right);

        // 坐标为p的元素已经放到了正确的位置
        // 左边快排
        randomQuickSort(nums,left,p-1);
        // 右边快排
        randomQuickSort(nums,p + 1,right);
    }
}

int parition(vector<int> &nums,int left,int right)
{
    int val = nums[left];
    int index = left;
    for(int i = left + 1; i <= right; i ++){
        if(nums[i] < val){
            index ++;
            if(index != i){
                swap(nums,i,index);
            }
        }
    }
    swap(nums,index,left);
    return index;
}

void quickSort(vector<int> &nums,int left,int right)
{
    if(left < right){
        auto p = parition(nums,left,right);
        
        quickSort(nums,left, p - 1);
        quickSort(nums,p + 1,right);
    }
}


int main()
{
    vector<int> nums;
    randomVector(nums,10);
    printVector(nums);
    auto  a = nums;

    quickSort(nums,0,nums.size() - 1);
    printVector(nums);

    srand(time(nullptr));

    printVector(a);
    randomQuickSort(a,0,a.size() - 1);
    printVector(a);

}