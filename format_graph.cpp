//
// Created by Yang on 2025/11/19.
//

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "The number of parameters is error!" << endl;
    cout << "Usage: ./format filename" << endl;
    return 1;
  }
  string filename = argv[1];
  fstream file;
  file.open(filename, ios::in);
  if (!file.is_open()) {
    cerr << "Failed to open the file " << filename << endl;
    return 2;
  }

  string line;
  map<int, int> vertices;
  map<string, int> labels;
  vector<map<int, set<int>>> edges; //标记边
  int vertices_num = 0, label_num = 0, labeled_edges_num = 0;
  while (getline(file, line)) {
    istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;

    // 以空格划分每一行，并存储到 tokens 向量中
    while (iss >> token) {
      tokens.push_back(token);
    }

    int u, v;
    string l;
    //如果是空行，跳过
    if (tokens.empty())
      continue;
    try {
      // 尝试读取三个变量
      u = stoi(tokens[0]);
      v = stoi(tokens[1]);
      l = tokens[2];
    } catch (const std::exception &e) {
      // 如果变量类型有错误，则跳过该行
      continue;
    }

    //去除自环边
    if (u == v)
      continue;

    //顶点检查
    auto it1 = vertices.insert({u, vertices_num});
    //插入成功，说明顶点不存在
    if (it1.second) {
      vertices_num++;
      // 创建一个空的 map<int, set<int>>
      map<int, set<int>> emptyMap;
      // 将空的 map 推入到 vector 中
      edges.push_back(emptyMap);
    }
    it1 = vertices.insert({v, vertices_num});
    //插入成功，说明顶点不存在
    if (it1.second) {
      vertices_num++;
      // 创建一个空的 map<int, set<int>>
      map<int, set<int>> emptyMap;
      // 将空的 map 推入到 vector 中
      edges.push_back(emptyMap);
    }

    //标签检查
    auto it2 = labels.insert({l, label_num});
    //若插入成功，说明标签不存在
    if (it2.second) {
      label_num++;
    }

    //插入边
    auto it3 = edges[vertices[u]][vertices[v]].insert(labels[l]);
    if (it3.second) {
      labeled_edges_num++;
    }
  }
  file.close();

  file.open(filename, ios::out);
  ofstream f(filename + "_mapping", ios::out);
  file << "vertices: " << vertices.size() << " topologyEdges: " << edges.size()
       << " labeledEdges: " << labeled_edges_num << " labels: " << labels.size()
       << endl;
  // The number of static edges, temporal edges and vertices.
  int uid = 0;
  for (auto &edge : edges) {
    for (const auto &e : edge) {
      int vid = e.first;
      auto it = e.second.begin();
      file << uid << " " << vid << " " << *it;
      ++it;
      while (it != e.second.end()) {
        file << "," << *it;
        ++it;
      }
    }
    file << endl;
    uid++;
  }
  f << "mapping relationships of vertices" << endl;
  for (auto &id : vertices) {
    f << id.first << " -> " << id.second << endl;
  }
  f << "mapping relationships of labels" << endl;
  for (auto &id : labels) {
    f << id.first << " -> " << id.second << endl;
  }
  file.close();
  f.close();

  return 0;
}