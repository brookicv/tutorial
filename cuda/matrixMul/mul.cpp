/*** 
 * @Author: Lq
 * @Date: 2022-01-07 10:13:13
 * @LastEditTime: 2022-01-07 14:47:09
 * @LastEditors: Lq
 * @Description: 矩阵乘法
 */
#include <random>
#include <iostream>
#include "timer.h"

using namespace std;

void randomMatrix(int *a,int row,int col)
{
    static random_device rd; // 产生随机数种子
    static mt19937 mt(rd());
    static uniform_int_distribution<int> gen(0,10);

    for(int i = 0; i < row; i ++){
        int *b = a + i * col;
        for(int j = 0; j < col; j ++){
            b[j] = gen(mt);
        }
    }
}

void printMatrix(int *a,int row,int col)
{
    for(int i = 0; i < row; i ++){
        int *b = a + i * col;
        for(int j = 0; j < col; j ++){
            cout << b[j] << " ";
        }
        cout << endl;
    }
}


int main()
{

    const int row = 1000;
    const int col = 1000;
    const int k = 1000;
    
    int a[row][col];
    randomMatrix((int*)a,row,k);
    
    int b[k][col];
    randomMatrix((int*)b,k,col);

    int c[row][col] = {0};

    Timer timer;

    for(int r = 0; r < row; r ++){
        for(int cl = 0; cl < col; cl ++){
            for(int j = 0; j < k; j ++){
                c[r][cl] += a[r][j] * b[j][cl];
            }
        }
    }

    auto duration = timer.elapsed();
    cout << duration << " ms" << endl;


    return 0;
}
