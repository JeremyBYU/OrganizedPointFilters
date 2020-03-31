#include <iostream>
#include "OrganizedPointFilters/OrganizedPointFilters.hpp"

int main(int argc, char const *argv[])
{
    std::string my_name = "Jeremy";
    std::cout << "My Name is " << my_name << std::endl;
    
    // Respond
    std::cout << OrganizedPointFilters::Hello(my_name) << std::endl;

}
