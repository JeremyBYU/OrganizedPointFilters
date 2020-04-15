#include <iostream>
#include "OrganizedPointFilters/OrganizedPointFilters.hpp"
#include "OrganizedPointFilters/Filter/Bilateral.hpp"

using namespace OrganizedPointFilters;

void InitRandom(Eigen::Ref<RowMatrixXVec3f> a)
{
    for (auto i = 0; i < a.rows(); ++i)
    {
        for (auto j = 0; j < a.cols(); ++j)
        {
            a(i,j) = Eigen::Vector3f::Random();
        }
    }
}

int main(int argc, char const *argv[])
{

    RowMatrixXVec3f a(250, 250);
    InitRandom(a);
    // std::cout << a.rows() << std::endl;
    int iterations = 5;
    std::cout << "Before" << std::endl;
    auto result = Filter::BilateralFilterNormals<3>(a, 1);
    std::cout << "Finished" << std::endl;

}
