/*** 
 * @Author: Lq
 * @Date: 2022-01-14 17:39:00
 * @LastEditTime: 2022-01-14 18:02:07
 * @LastEditors: Lq
 * @Description: 构造函数
 */

#include <cstdlib>
#include <cstring>

class CMyString
{

public:
    CMyString(char *pdata = nullptr);
    ~CMyString();

    CMyString(const CMyString &str);
    // 赋值运算符
    CMyString& operator =(const CMyString &str);

private:
    char *pdata;

};

// 拷贝构造函数
CMyString::CMyString(const CMyString &str)
{
    // 资源还没有分配，手动分配资源，再拷贝传入对象的数据
    pdata = new char[strlen(str.pdata) + 1];
    strcpy(pdata,str.pdata);
}

// 赋值运算符
// 对象实例已经存在，将传入的对象值拷贝过来
CMyString& CMyString::operator=(const CMyString &str)
{
    if(this == &str){
        return *this;
    }

    // 删除自己的已占用的空间
    delete[] pdata;

    // 重新分配数据
    pdata = new char[strlen(str.pdata) + 1];
    strcpy(pdata,str.pdata);

    return *this;
}
