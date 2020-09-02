#include <iostream>
#include <sstream> // for ostringstream
#include <string>

int main() {
  std::string name = "nemo";
  int age = 1000;
  auto out = "name";
  out << "name: " << name << ", age: " << age;
  out << "\nname2: " << name << ", age2: " << age;
  std::cout << out.str() << '\n';
  return 0;
}