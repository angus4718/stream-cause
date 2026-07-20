#include "market_event.hpp"

namespace sc {

const std::unordered_map<std::string, uint32_t> SYMBOL_TO_ID = {
    {"AAPL", 0}, {"MSFT", 1}, {"NVDA", 2}, {"AMZN", 3}, {"GOOGL", 4},
    {"META", 5}, {"TSLA", 6}, {"AVGO", 7}, {"COST", 8}, {"ADBE", 9},
    {"ASML", 10}, {"AMD", 11}, {"QCOM", 12}, {"TXN", 13}, {"INTU", 14},
    {"AMAT", 15}, {"MU", 16}, {"LRCX", 17}, {"KLAC", 18}, {"MRVL", 19},
    {"PANW", 20}, {"CDNS", 21}, {"SNPS", 22}, {"NXPI", 23}, {"ADI", 24},
    {"MCHP", 25}, {"FTNT", 26}, {"REGN", 27}, {"BIIB", 28}, {"VRTX", 29},
    {"GILD", 30}, {"IDXX", 31}, {"ISRG", 32}, {"ALGN", 33}, {"DXCM", 34},
    {"ILMN", 35}, {"SGEN", 36}, {"MRNA", 37}, {"PYPL", 38}, {"EBAY", 39},
    {"NFLX", 40}, {"CMCSA", 41}, {"CHTR", 42}, {"ATVI", 43}, {"EA", 44},
    {"TTWO", 45}, {"WBA", 46}, {"FAST", 47}, {"ODFL", 48}, {"PAYX", 49},
    {"SPY", 50}, {"QQQ", 51}, {"IWM", 52},
    {"ES.c.0", 53}, {"NQ.c.0", 54}, {"ZN.c.0", 55},
    {"ZB.c.0", 56}, {"6E.c.0", 57}, {"CL.c.0", 58},
};

const std::unordered_map<uint32_t, std::string> ID_TO_SYMBOL = {
    {0, "AAPL"}, {1, "MSFT"}, {2, "NVDA"}, {3, "AMZN"}, {4, "GOOGL"},
    {5, "META"}, {6, "TSLA"}, {7, "AVGO"}, {8, "COST"}, {9, "ADBE"},
    {10, "ASML"}, {11, "AMD"}, {12, "QCOM"}, {13, "TXN"}, {14, "INTU"},
    {15, "AMAT"}, {16, "MU"}, {17, "LRCX"}, {18, "KLAC"}, {19, "MRVL"},
    {20, "PANW"}, {21, "CDNS"}, {22, "SNPS"}, {23, "NXPI"}, {24, "ADI"},
    {25, "MCHP"}, {26, "FTNT"}, {27, "REGN"}, {28, "BIIB"}, {29, "VRTX"},
    {30, "GILD"}, {31, "IDXX"}, {32, "ISRG"}, {33, "ALGN"}, {34, "DXCM"},
    {35, "ILMN"}, {36, "SGEN"}, {37, "MRNA"}, {38, "PYPL"}, {39, "EBAY"},
    {40, "NFLX"}, {41, "CMCSA"}, {42, "CHTR"}, {43, "ATVI"}, {44, "EA"},
    {45, "TTWO"}, {46, "WBA"}, {47, "FAST"}, {48, "ODFL"}, {49, "PAYX"},
    {50, "SPY"}, {51, "QQQ"}, {52, "IWM"},
    {53, "ES.c.0"}, {54, "NQ.c.0"}, {55, "ZN.c.0"},
    {56, "ZB.c.0"}, {57, "6E.c.0"}, {58, "CL.c.0"},
};

}  // namespace sc
