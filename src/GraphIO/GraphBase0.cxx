/**
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date June, 2017
 * @version v1.3
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 */
#include "GraphIO/GraphBase.hpp"
#include "Support/Host/Basic.hpp"   //WARNING
#include "Support/Host/FileUtil.hpp"//xlib::file_size
#include "Support/Host/Numeric.hpp" //xlib::check_overflow
#include <iostream>                 //std::cout
#include <sstream>                  //std::istringstream

namespace graph {

template<typename vid_t, typename eoff_t>
void GraphBase<vid_t, eoff_t>::read(const char* filename,
                                    const Property& prop) {
    xlib::check_regular_file(filename);
    size_t size = xlib::file_size(filename);
    _graph_name = xlib::extract_filename(filename);
    _prop       = prop;

    if (prop.is_print()) {
        std::cout << "\nGraph File:\t" << _graph_name
                  << "       Size: " <<  xlib::format(size / xlib::MB) << " MB";
    }

    std::string file_ext = xlib::extract_file_extension(filename);
    if (file_ext == ".bin") {
        if (prop.is_print())
            std::cout << "            (Binary)Reading...   \n";
        if (prop.is_print() && (prop.is_randomize() || prop.is_sort()))
            std::cerr << "#input sort/randomize ignored on binary format\n";
        readBinary(filename, prop.is_print());
        return;
    }

    std::ifstream fin;
    //IO improvements START ----------------------------------------------------
    const int BUFFER_SIZE = 1 * xlib::MB;
    char buffer[BUFFER_SIZE];
    //std::ios_base::sync_with_stdio(false);
    fin.tie(nullptr);
    fin.rdbuf()->pubsetbuf(buffer, BUFFER_SIZE);
    //IO improvements END ------------------------------------------------------
    fin.open(filename);
    std::string first_str;
    fin >> first_str;
    fin.seekg(std::ios::beg);

    if (file_ext == ".mtx" && first_str == "%%MatrixMarket") {
        if (prop.is_print())
            std::cout << "      (MatrixMarket)\nReading...   ";
        readMarket(fin, prop.is_print());
    }
    else if (file_ext == ".graph") {
        if (prop.is_print())
            std::cout << "        (Dimacs10th)\nReading...   ";
        if (prop.is_randomize() || prop.is_sort()) {
            std::cerr << "#input sort/randomize ignored on Dimacs10th format"
                      << std::endl;
        }
        readDimacs10(fin, prop.is_print());
    }
    else if (file_ext == ".gr" && (first_str == "c"|| first_str == "p")) {
        if (prop.is_print())
            std::cout << "         (Dimacs9th)\nReading...   ";
        readDimacs9(fin, prop.is_print());
    }
    else if (file_ext == ".txt" && first_str == "#") {
        if (prop.is_print())
            std::cout << "              (SNAP)\nReading...   ";
        readSnap(fin, prop.is_print());
    }
    else if (file_ext == ".edges") {
        if (prop.is_print())
            std::cout << "    (Net Repository)\nReading...   ";
        readNetRepo(fin, prop.is_print());
    }
    else if (first_str == "%") {
        if (prop.is_print())
            std::cout << "            (Konect)\nReading...   ";
        readKonect(fin, prop.is_print());
    } else
        ERROR("Graph type not recognized");
    fin.close();
    COOtoCSR();
}

//==============================================================================

template<typename vid_t, typename eoff_t>
GInfo GraphBase<vid_t, eoff_t>::getMarketHeader(std::ifstream& fin) {
    std::string header_lines;
    std::getline(fin, header_lines);
    auto direction = header_lines.find("symmetric") != std::string::npos ?
                        Structure::UNDIRECTED : Structure::DIRECTED;
    _directed_to_undirected = direction == Structure::UNDIRECTED;

    if (header_lines.find("integer") != std::string::npos)
        _structure._wtype = Structure::INTEGER;
    if (header_lines.find("real") != std::string::npos)
        _structure._wtype = Structure::REAL;

    while (fin.peek() == '%')
        xlib::skip_lines(fin);

    size_t rows, columns, num_lines;
    fin >> rows >> columns >> num_lines;
    if (rows != columns)
        WARNING("Rectangular matrix");
    xlib::skip_lines(fin);
    size_t num_edges = direction == Structure::UNDIRECTED ? num_lines * 2 :
                                                          : num_lines;
    return { std::max(rows, columns), num_edges, num_lines, direction };
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
GInfo GraphBase<vid_t, eoff_t>::getDimacs9Header(std::ifstream& fin) {
    while (fin.peek() == 'c')
        xlib::skip_lines(fin);

    xlib::skip_words(fin, 2);
    size_t num_vertices, num_edges;
    fin >> num_vertices >> num_edges;
    return { num_vertices, num_edges, num_edges, Structure::UNDIRECTED };
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
GInfo GraphBase<vid_t, eoff_t>::getDimacs10Header(std::ifstream& fin) {
    while (fin.peek() == '%')
        xlib::skip_lines(fin);

    size_t num_vertices, num_edges;
    fin >> num_vertices >> num_edges;
    Structure::Enum direction;

    if (fin.peek() == '\n')
        direction = Structure::UNDIRECTED;
    else {
        std::string flag;
        fin >> flag;
        direction = flag == "100" ? Structure::DIRECTED : Structure::UNDIRECTED;
        xlib::skip_lines(fin);
    }
    _directed_to_undirected = direction == Structure::UNDIRECTED;
    if (direction == Structure::UNDIRECTED)
        num_edges *= 2;
    return { num_vertices, num_edges, num_vertices, direction };
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
GInfo GraphBase<vid_t, eoff_t>::getKonectHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    auto direction = (str == "asym") || (str == "bip") ?
                        Structure::DIRECTED : Structure::UNDIRECTED;
    size_t num_edges, value1, value2;
    fin >> str >> num_edges >> value1 >> value2;
    xlib::skip_lines(fin);
    if (str != "%")
        ERROR("Wrong file format")
    _directed_to_undirected = direction == Structure::UNDIRECTED;
    return { std::max(value1, value2), num_edges, direction };
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
void GraphBase<vid_t, eoff_t>::getNetRepoHeader(std::ifstream& fin) {
    std::string str;
    fin >> str >> str;
    //_header_direction = (str == "directed") ? Structure::DIRECTED
    //                                        : Structure::UNDIRECTED;
    xlib::skip_lines(fin);
}

//------------------------------------------------------------------------------

template<typename vid_t, typename eoff_t>
GInfo GraphBase<vid_t, eoff_t>::getSnapHeader(std::ifstream& fin) {
    std::string tmp;
    fin >> tmp >> tmp;
    Structure::Enum direction = (tmp == "Undirected") ? Structure::UNDIRECTED
                                                      : Structure::DIRECTED;
    xlib::skip_lines(fin);

    size_t num_lines = 0, num_vertices = 0;
    while (fin.peek() == '#') {
        std::getline(fin, tmp);
        if (tmp.substr(2, 6) == "Nodes:") {
            std::istringstream stream(tmp);
            stream >> tmp >> tmp >> num_vertices >> tmp >> num_lines;
            break;
        }
    }
    xlib::skip_lines(fin);
    _directed_to_undirected = direction == Structure::UNDIRECTED;
    return { num_vertices, num_lines, direction };
}

//------------------------------------------------------------------------------

template class GraphBase<int, int>;
template class GraphBase<int64_t, int64_t>;

} // namespace graph