#ifndef TOOL_TYPE_H
#define TOOL_TYPE_H

typedef enum {
    CODE_CHECK = 0,
    APP_METRICE = 1,
    MEM_TRACE = 2,
    HOT_ANALYSIS = 3,
    UVM_ADVISOR = 4,
    APP_ANALYSIS = 5,
    APP_ANALYSIS_CPU = 6,
    TOOL_NUMS = 7
} AnalysisTool_t;

#endif // TOOL_TYPE_H