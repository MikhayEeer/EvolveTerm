~/Desktop/fox3/TermDB/EvolveTerm$ evolveterm extract --input examples/llm_nested_term_samples/generated1229program1/program.c 
[Debug] Loaded config with tag: default, 
[Debug]--llm config:-- 
{'provider': 'qwen', 'baseurl': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'api_key': 'sk-667378412e61493c8e4594bcebe28dff', 'model': 'qwen3-max', 'payload_template': {'max_tokens': 2048, 'temperature': 0.0}, 'pricing_per_millon_tokens_RMB': 16, 'tags': ['default', 'better'], 'context_window': '256k'}
[Debug]--print end--


Extracting loops from examples/llm_nested_term_samples/generated1229program1/program.c...
[Debug][Extractor] Using LLM extraction method, and Successfully get verified loops.
[
  "for(temp = 0; temp < 100; temp++) memo = 0;",
  "for (i = 1; i < 100 && j > 10; i = i * 3 + 1, j = j / 2) {\n\tidx = (i + j) % 100;\n\tmemo = (i << 2) ^ (j >> 
1);\n\tsink += memo;\n\twhile (k < 100 && l > 0) {\n\t\tp = &memo;\n\t\tif (*p > 100) {\n\t\t\tsink -= 1;\n\t\t} 
else {\n\t\t\tsink += 1;\n\t\t}\n\t\tdo {\n\t\t\tval = (k * k) % 50;\n\t\t\tmemo = l + m;\n\t\t\ttemp = (m > 5) ?
10 : 20;\n\t\t\tsink = sink ^ temp;\n\t\t\tfor (l = 50; l > 0 && m > 0; l = l / 2) {\n\t\t\t\tint noise = (i * 
100) + j - k;\n\t\t\t\tprintf(\"Example 1: Looping (noise:%d)\\n\", noise);\n\t\t\t\tsink += 
noise;\n\t\t\t}\n\t\t\tsink = sink + k - l;\n\t\t\tk = k * 2;\n\t\t} while (k < 100 && m > 0);\n\t\tidx = (j * 2)
% 100;\n\t\tmemo = j;\n\t\tj = j - 1;\n\t}\n}",
  "while (k < 100 && l > 0) {\n\tp = &memo;\n\tif (*p > 100) {\n\t\tsink -= 1;\n\t} else {\n\t\tsink += 
1;\n\t}\n\tdo {\n\t\tval = (k * k) % 50;\n\t\tmemo = l + m;\n\t\ttemp = (m > 5) ? 10 : 20;\n\t\tsink = sink ^ 
temp;\n\t\tfor (l = 50; l > 0 && m > 0; l = l / 2) {\n\t\t\tint noise = (i * 100) + j - 
k;\n\t\t\tprintf(\"Example 1: Looping (noise:%d)\\n\", noise);\n\t\t\tsink += noise;\n\t\t}\n\t\tsink = sink + k 
- l;\n\t\tk = k * 2;\n\t} while (k < 100 && m > 0);\n\tidx = (j * 2) % 100;\n\tmemo = j;\n\tj = j - 1;\n}",
  "do {\n\tval = (k * k) % 50;\n\tmemo = l + m;\n\ttemp = (m > 5) ? 10 : 20;\n\tsink = sink ^ temp;\n\tfor (l = 
50; l > 0 && m > 0; l = l / 2) {\n\t\tint noise = (i * 100) + j - k;\n\t\tprintf(\"Example 1: Looping 
(noise:%d)\\n\", noise);\n\t\tsink += noise;\n\t}\n\tsink = sink + k - l;\n\tk = k * 2;\n} while (k < 100 && m > 
0);",
  "for (l = 50; l > 0 && m > 0; l = l / 2) {\n\tint noise = (i * 100) + j - k;\n\tprintf(\"Example 1: Looping 
(noise:%d)\\n\", noise);\n\tsink += noise;\n}"
]