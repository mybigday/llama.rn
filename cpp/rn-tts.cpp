#include "rn-tts.h"
#include "rn-llama.h"
#include "anyascii.h"
#include "common.h"
#include <regex>
#include <map>
#include <sstream>
#include <iomanip>
#include <codecvt>
#include <locale>
#include <thread>
#include <cmath>

#ifdef _WIN32
  #define M_PI 3.14159265358979323846
#endif

namespace rnllama {

// Constants definitions
const std::string default_audio_text = "<|text_start|>the<|text_sep|>overall<|text_sep|>package<|text_sep|>from<|text_sep|>just<|text_sep|>two<|text_sep|>people<|text_sep|>is<|text_sep|>pretty<|text_sep|>remarkable<|text_sep|>sure<|text_sep|>i<|text_sep|>have<|text_sep|>some<|text_sep|>critiques<|text_sep|>about<|text_sep|>some<|text_sep|>of<|text_sep|>the<|text_sep|>gameplay<|text_sep|>aspects<|text_sep|>but<|text_sep|>its<|text_sep|>still<|text_sep|>really<|text_sep|>enjoyable<|text_sep|>and<|text_sep|>it<|text_sep|>looks<|text_sep|>lovely<|text_sep|>";

const std::string default_audio_data = R"(<|audio_start|>
the<|t_0.08|><|code_start|><|257|><|740|><|636|><|913|><|788|><|1703|><|code_end|>
overall<|t_0.36|><|code_start|><|127|><|201|><|191|><|774|><|700|><|532|><|1056|><|557|><|798|><|298|><|1741|><|747|><|1662|><|1617|><|1702|><|1527|><|368|><|1588|><|1049|><|1008|><|1625|><|747|><|1576|><|728|><|1019|><|1696|><|1765|><|code_end|>
package<|t_0.56|><|code_start|><|935|><|584|><|1319|><|627|><|1016|><|1491|><|1344|><|1117|><|1526|><|1040|><|239|><|1435|><|951|><|498|><|723|><|1180|><|535|><|789|><|1649|><|1637|><|78|><|465|><|1668|><|901|><|595|><|1675|><|117|><|1009|><|1667|><|320|><|840|><|79|><|507|><|1762|><|1508|><|1228|><|1768|><|802|><|1450|><|1457|><|232|><|639|><|code_end|>
from<|t_0.19|><|code_start|><|604|><|782|><|1682|><|872|><|1532|><|1600|><|1036|><|1761|><|647|><|1554|><|1371|><|653|><|1595|><|950|><|code_end|>
just<|t_0.25|><|code_start|><|1782|><|1670|><|317|><|786|><|1748|><|631|><|599|><|1155|><|1364|><|1524|><|36|><|1591|><|889|><|1535|><|541|><|440|><|1532|><|50|><|870|><|code_end|>
two<|t_0.24|><|code_start|><|1681|><|1510|><|673|><|799|><|805|><|1342|><|330|><|519|><|62|><|640|><|1138|><|565|><|1552|><|1497|><|1552|><|572|><|1715|><|1732|><|code_end|>
people<|t_0.39|><|code_start|><|593|><|274|><|136|><|740|><|691|><|633|><|1484|><|1061|><|1138|><|1485|><|344|><|428|><|397|><|1562|><|645|><|917|><|1035|><|1449|><|1669|><|487|><|442|><|1484|><|1329|><|1832|><|1704|><|600|><|761|><|653|><|269|><|code_end|>
is<|t_0.16|><|code_start|><|566|><|583|><|1755|><|646|><|1337|><|709|><|802|><|1008|><|485|><|1583|><|652|><|10|><|code_end|>
pretty<|t_0.32|><|code_start|><|1818|><|1747|><|692|><|733|><|1010|><|534|><|406|><|1697|><|1053|><|1521|><|1355|><|1274|><|816|><|1398|><|211|><|1218|><|817|><|1472|><|1703|><|686|><|13|><|822|><|445|><|1068|><|code_end|>
remarkable<|t_0.68|><|code_start|><|230|><|1048|><|1705|><|355|><|706|><|1149|><|1535|><|1787|><|1356|><|1396|><|835|><|1583|><|486|><|1249|><|286|><|937|><|1076|><|1150|><|614|><|42|><|1058|><|705|><|681|><|798|><|934|><|490|><|514|><|1399|><|572|><|1446|><|1703|><|1346|><|1040|><|1426|><|1304|><|664|><|171|><|1530|><|625|><|64|><|1708|><|1830|><|1030|><|443|><|1509|><|1063|><|1605|><|1785|><|721|><|1440|><|923|><|code_end|>
sure<|t_0.36|><|code_start|><|792|><|1780|><|923|><|1640|><|265|><|261|><|1525|><|567|><|1491|><|1250|><|1730|><|362|><|919|><|1766|><|543|><|1|><|333|><|113|><|970|><|252|><|1606|><|133|><|302|><|1810|><|1046|><|1190|><|1675|><|code_end|>
i<|t_0.08|><|code_start|><|123|><|439|><|1074|><|705|><|1799|><|637|><|code_end|>
have<|t_0.16|><|code_start|><|1509|><|599|><|518|><|1170|><|552|><|1029|><|1267|><|864|><|419|><|143|><|1061|><|0|><|code_end|>
some<|t_0.16|><|code_start|><|619|><|400|><|1270|><|62|><|1370|><|1832|><|917|><|1661|><|167|><|269|><|1366|><|1508|><|code_end|>
critiques<|t_0.60|><|code_start|><|559|><|584|><|1163|><|1129|><|1313|><|1728|><|721|><|1146|><|1093|><|577|><|928|><|27|><|630|><|1080|><|1346|><|1337|><|320|><|1382|><|1175|><|1682|><|1556|><|990|><|1683|><|860|><|1721|><|110|><|786|><|376|><|1085|><|756|><|1523|><|234|><|1334|><|1506|><|1578|><|659|><|612|><|1108|><|1466|><|1647|><|308|><|1470|><|746|><|556|><|1061|><|code_end|>
about<|t_0.29|><|code_start|><|26|><|1649|><|545|><|1367|><|1263|><|1728|><|450|><|859|><|1434|><|497|><|1220|><|1285|><|179|><|755|><|1154|><|779|><|179|><|1229|><|1213|><|922|><|1774|><|1408|><|code_end|>
some<|t_0.23|><|code_start|><|986|><|28|><|1649|><|778|><|858|><|1519|><|1|><|18|><|26|><|1042|><|1174|><|1309|><|1499|><|1712|><|1692|><|1516|><|1574|><|code_end|>
of<|t_0.07|><|code_start|><|197|><|716|><|1039|><|1662|><|64|><|code_end|>
the<|t_0.08|><|code_start|><|1811|><|1568|><|569|><|886|><|1025|><|1374|><|code_end|>
gameplay<|t_0.48|><|code_start|><|1269|><|1092|><|933|><|1362|><|1762|><|1700|><|1675|><|215|><|781|><|1086|><|461|><|838|><|1022|><|759|><|649|><|1416|><|1004|><|551|><|909|><|787|><|343|><|830|><|1391|><|1040|><|1622|><|1779|><|1360|><|1231|><|1187|><|1317|><|76|><|997|><|989|><|978|><|737|><|189|><|code_end|>
aspects<|t_0.56|><|code_start|><|1423|><|797|><|1316|><|1222|><|147|><|719|><|1347|><|386|><|1390|><|1558|><|154|><|440|><|634|><|592|><|1097|><|1718|><|712|><|763|><|1118|><|1721|><|1311|><|868|><|580|><|362|><|1435|><|868|><|247|><|221|><|886|><|1145|><|1274|><|1284|><|457|><|1043|><|1459|><|1818|><|62|><|599|><|1035|><|62|><|1649|><|778|><|code_end|>
but<|t_0.20|><|code_start|><|780|><|1825|><|1681|><|1007|><|861|><|710|><|702|><|939|><|1669|><|1491|><|613|><|1739|><|823|><|1469|><|648|><|code_end|>
its<|t_0.09|><|code_start|><|92|><|688|><|1623|><|962|><|1670|><|527|><|599|><|code_end|>
still<|t_0.27|><|code_start|><|636|><|10|><|1217|><|344|><|713|><|957|><|823|><|154|><|1649|><|1286|><|508|><|214|><|1760|><|1250|><|456|><|1352|><|1368|><|921|><|615|><|5|><|code_end|>
really<|t_0.36|><|code_start|><|55|><|420|><|1008|><|1659|><|27|><|644|><|1266|><|617|><|761|><|1712|><|109|><|1465|><|1587|><|503|><|1541|><|619|><|197|><|1019|><|817|><|269|><|377|><|362|><|1381|><|507|><|1488|><|4|><|1695|><|code_end|>
enjoyable<|t_0.49|><|code_start|><|678|><|501|><|864|><|319|><|288|><|1472|><|1341|><|686|><|562|><|1463|><|619|><|1563|><|471|><|911|><|730|><|1811|><|1006|><|520|><|861|><|1274|><|125|><|1431|><|638|><|621|><|153|><|876|><|1770|><|437|><|987|><|1653|><|1109|><|898|><|1285|><|80|><|593|><|1709|><|843|><|code_end|>
and<|t_0.15|><|code_start|><|1285|><|987|><|303|><|1037|><|730|><|1164|><|502|><|120|><|1737|><|1655|><|1318|><|code_end|>
it<|t_0.09|><|code_start|><|848|><|1366|><|395|><|1601|><|1513|><|593|><|1302|><|code_end|>
looks<|t_0.27|><|code_start|><|1281|><|1266|><|1755|><|572|><|248|><|1751|><|1257|><|695|><|1380|><|457|><|659|><|585|><|1315|><|1105|><|1776|><|736|><|24|><|736|><|654|><|1027|><|code_end|>
lovely<|t_0.56|><|code_start|><|634|><|596|><|1766|><|1556|><|1306|><|1285|><|1481|><|1721|><|1123|><|438|><|1246|><|1251|><|795|><|659|><|1381|><|1658|><|217|><|1772|><|562|><|952|><|107|><|1129|><|1112|><|467|><|550|><|1079|><|840|><|1615|><|1469|><|1380|><|168|><|917|><|836|><|1827|><|437|><|583|><|67|><|595|><|1087|><|1646|><|1493|><|1677|><|code_end|>)";

const char *OUTETTS_V1_GRAMMAR = R"(
root       ::= NL? wordAudioBlock+ audioEnd NL eos?
wordAudioBlock ::= WORD codeBlock NL
codeBlock ::= TIME CODE*
 eos      ::= "<|im_end|>"
codeStart ::= "<|code_start|>"
codeEnd ::= "<|code_end|>"
audioEnd   ::= "<|audio_end|>"
WORD       ::= [A-Za-z]+
NL         ::= [\n]
TIME  ::= "<|t_" DECIMAL "|>"
CODE    ::= "<|" DIGITS "|>"
DIGITS     ::= [0-9]+
DECIMAL    ::= [0-9]+ "." [0-9]+
)";

const char *OUTETTS_V2_GRAMMAR = R"(
root       ::= NL? content+ audioEnd NL eos?
content ::= wordAudioBlock | emotionBlock
wordAudioBlock ::= WORD punch* codeBlock space NL
codeBlock ::= TIME CODE*
emotionBlock ::= emotionStart TEXT emotionEnd space NL
TEXT ::= [A-Za-z0-9 .,?!]+
 eos      ::= "<|im_end|>"
emotionStart ::= "<|emotion_start|>"
emotionEnd ::= "<|emotion_end|>"
audioEnd   ::= "<|audio_end|>"
space      ::= "<|space|>"
WORD       ::= [A-Za-z]+
NL         ::= [\n]
TIME  ::= "<|t_" DECIMAL "|>"
CODE    ::= "<|" DIGITS "|>"
DIGITS     ::= [0-9]+
DECIMAL    ::= [0-9]+ "." [0-9]+
punch ::= "<|" [a-z_]+ "|>"
)";

// Number conversion maps and functions
static const std::map<int, std::string> ones = {
    {0, "zero"}, {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"},
    {5, "five"}, {6, "six"}, {7, "seven"}, {8, "eight"}, {9, "nine"},
    {10, "ten"}, {11, "eleven"}, {12, "twelve"}, {13, "thirteen"}, {14, "fourteen"},
    {15, "fifteen"}, {16, "sixteen"}, {17, "seventeen"}, {18, "eighteen"}, {19, "nineteen"}
};

static const std::map<int, std::string> tens = {
    {2, "twenty"}, {3, "thirty"}, {4, "forty"}, {5, "fifty"},
    {6, "sixty"}, {7, "seventy"}, {8, "eighty"}, {9, "ninety"}
};

// Convert a number less than 1000 to words
static std::string convert_less_than_thousand(int num) {
    std::string result;

    if (num >= 100) {
        result += ones.at(num / 100) + " hundred ";
        num %= 100;
    }

    if (num >= 20) {
        result += tens.at(num / 10);
        if (num % 10 > 0) {
            result += "-" + ones.at(num % 10);
        }
    } else if (num > 0) {
        result += ones.at(num);
    }

    return result;
}

std::string number_to_words(const std::string & number_str) {
    try {
        size_t decimal_pos = number_str.find('.');
        std::string integer_part = number_str.substr(0, decimal_pos);

        int int_number = std::stoi(integer_part);
        std::string result;

        if (int_number == 0) {
            result = "zero";
        } else {
            if (int_number >= 1000000000) {
                int billions = int_number / 1000000000;
                result += convert_less_than_thousand(billions) + " billion ";
                int_number %= 1000000000;
            }

            if (int_number >= 1000000) {
                int millions = int_number / 1000000;
                result += convert_less_than_thousand(millions) + " million ";
                int_number %= 1000000;
            }

            if (int_number >= 1000) {
                int thousands = int_number / 1000;
                result += convert_less_than_thousand(thousands) + " thousand ";
                int_number %= 1000;
            }

            if (int_number > 0) {
                result += convert_less_than_thousand(int_number);
            }
        }

        // Handle decimal part
        if (decimal_pos != std::string::npos) {
            result += " point";
            std::string decimal_part = number_str.substr(decimal_pos + 1);
            for (char digit : decimal_part) {
                result += " " + ones.at(digit - '0');
            }
        }

        return result;
    } catch (const std::exception& e) {
        // Skip if fails
        return " ";
    }
}

std::string replace_numbers_with_words(const std::string & input_text) {
    std::regex number_pattern(R"(\d+(\.\d+)?)");
    std::string result;
    auto it = std::sregex_iterator(input_text.begin(), input_text.end(), number_pattern);
    auto end = std::sregex_iterator();

    size_t last_pos = 0;
    for (std::sregex_iterator i = it; i != end; ++i) {
        const std::smatch& match = *i;
        result.append(input_text, last_pos, match.position() - last_pos);
        result.append(number_to_words(match.str()));
        last_pos = match.position() + match.length();
    }
    result.append(input_text, last_pos);

    return result;
}

static std::string anyascii_string(const std::string &input) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    auto wstr = converter.from_bytes(input);
    std::string output;
    for (char32_t c : wstr) {
        const char *r;
        size_t rlen = anyascii(c, &r);
        output.append(r, rlen);
    }
    return output;
}

std::string process_text(const std::string & text, const tts_type tts_type) {
    std::string processed_text = replace_numbers_with_words(text);

    if (tts_type == OUTETTS_V0_2 || tts_type == OUTETTS_V0_3) {
        processed_text = anyascii_string(processed_text);

        std::regex dashes(R"([—–-])");
        processed_text = std::regex_replace(processed_text, dashes, " ");
    }

    std::transform(processed_text.begin(), processed_text.end(),
                  processed_text.begin(), ::tolower);

    std::regex special_chars(R"([-_/,\.\\])");
    processed_text = std::regex_replace(processed_text, special_chars, " ");

    std::regex non_alpha(R"([^a-z\s])");
    processed_text = std::regex_replace(processed_text, non_alpha, "");

    std::regex multiple_spaces(R"(\s+)");
    processed_text = std::regex_replace(processed_text, multiple_spaces, " ");

    processed_text = std::regex_replace(processed_text, std::regex(R"(^\s+|\s+$)"), "");

    std::string separator = (tts_type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
    processed_text = std::regex_replace(processed_text, std::regex(R"(\s)"), separator);

    return processed_text;
}

std::string audio_text_from_speaker(json speaker, const tts_type type) {
    std::string audio_text = "<|text_start|>";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string separator = (type == OUTETTS_V0_3) ? "<|space|>" : "<|text_sep|>";
        for (const auto &word : speaker["words"]) {
            audio_text += word["word"].get<std::string>() + separator;
        }
    }

    return audio_text;
}

std::string audio_data_from_speaker(json speaker, const tts_type type) {
    std::string audio_data = "<|audio_start|>\n";

    if (type == OUTETTS_V0_2 || type == OUTETTS_V0_3) {
        std::string code_start = (type == OUTETTS_V0_3) ? "" : "<|code_start|>";
        std::string code_end = (type == OUTETTS_V0_3) ? "<|space|>" : "<|code_end|>";
        for (const auto &word : speaker["words"]) {
            std::string word_text = word["word"].get<std::string>();
            double duration = word["duration"].get<double>();
            std::vector<int> codes = word["codes"].get<std::vector<int>>();

            // Create the audio output entry
            std::ostringstream word_entry;
            word_entry << word_text << "<|t_" << std::fixed << std::setprecision(2)
                       << duration << "|>" + code_start;
            for (const auto &Code : codes) {
                word_entry << "<|" << Code << "|>";
            }
            word_entry << code_end << "\n";
            audio_data += word_entry.str();
        }
    }

    return audio_data;
}

// Constructor and destructor implementations
llama_rn_context_tts::llama_rn_context_tts(const std::string &vocoder_model_path, int batch_size) {
  common_params vocoder_params;
  vocoder_params.model.path = vocoder_model_path;
  vocoder_params.embedding = true;
  vocoder_params.ctx_shift = false;
  if (batch_size > 0) {
      vocoder_params.n_batch = batch_size;
  }
  vocoder_params.n_ubatch = vocoder_params.n_batch;

  init_result = common_init_from_params(vocoder_params);
  params = vocoder_params;
  model = init_result.model.get();
  ctx = init_result.context.get();

  if (model == nullptr || ctx == nullptr) {
      LOG_ERROR("Failed to load vocoder model: %s", vocoder_model_path.c_str());
      throw std::runtime_error("Failed to load vocoder model");
  }
  type = UNKNOWN; // Will be determined when used
}

llama_rn_context_tts::~llama_rn_context_tts() {
  // init_result will handle cleanup automatically when it goes out of scope
  model = nullptr;
  ctx = nullptr;
  type = UNKNOWN;
}

void llama_rn_context_tts::setGuideTokens(const std::vector<llama_token> &tokens) {
    guide_tokens = tokens;
}

// Audio processing functions - FFT and related utilities
static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = -1;
    if (periodic) {
        offset = 0;
    }
    for (int i = 0; i < length; i++) {
        output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

static void twiddle(float * real, float * imag, int k, int N) {
    float angle = 2 * M_PI * k / N;
    *real = cos(angle);
    *imag = sin(angle);
}

static void irfft(int n, const float * inp_cplx, float * out_real) {
    int N = n / 2 + 1;

    std::vector<float> real_input(N);
    std::vector<float> imag_input(N);
    for (int i = 0; i < N; ++i) {
        real_input[i] = inp_cplx[2 * i];
        imag_input[i] = inp_cplx[2 * i + 1];
    }

    std::vector<float> real_output(n);
    std::vector<float> imag_output(n);

    for (int k = 0; k < n; ++k) {
        real_output[k] = 0.0f;
        imag_output[k] = 0.0f;
        for (int m = 0; m < N; ++m) {
            float twiddle_real;
            float twiddle_imag;

            twiddle(&twiddle_real, &twiddle_imag, k * m, n);

            real_output[k] += real_input[m] * twiddle_real - imag_input[m] * twiddle_imag;
            imag_output[k] += real_input[m] * twiddle_imag + imag_input[m] * twiddle_real;
        }
    }

    for (int i = 0; i < n; ++i) {
        out_real[i] = real_output[i] / N;
    }
}

static void fold(const std::vector<float> & data, int64_t n_out, int64_t n_win, int64_t n_hop, int64_t n_pad, std::vector<float> & output) {
    int64_t output_height = n_out;
    int64_t kernel_w = n_win;
    int64_t stride_w = n_hop;
    int64_t width    = n_out;

    output.resize(width, 0.0f);

    int64_t col_idx = 0;
    for (int64_t w_col = 0; w_col < width; ++w_col) {
        int64_t start = w_col * stride_w - n_pad;
        int64_t end   = start + kernel_w;

        for (int64_t w_im = start; w_im < end; ++w_im) {
            if (w_im >= 0 && w_im < output_height && col_idx < (int64_t) data.size()) {
                output[w_im] += data[col_idx];
            }
            col_idx++;
        }
    }

    output.resize(n_out - 2 * n_pad);
}

std::vector<float> embd_to_audio(
        const float * embd,
        const int n_codes,
        const int n_embd,
        const int n_thread) {
    const int n_fft = 1280;
    const int n_hop = 320;
    const int n_win = 1280;
    const int n_pad = (n_win - n_hop)/2;
    const int n_out = (n_codes - 1)*n_hop + n_win;

    std::vector<float> hann(n_fft);

    fill_hann_window(hann.size(), true, hann.data());

    int n_spec = n_embd*n_codes;

    std::vector<float> E (n_spec);
    std::vector<float> S (n_spec);
    std::vector<float> ST(n_spec);

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd; ++k) {
            E[k*n_codes + l] = embd[l*n_embd + k];
        }
    }

    for (int k = 0; k < n_embd/2; ++k) {
        for (int l = 0; l < n_codes; ++l) {
            float mag = E[(k           )*n_codes + l];
            float phi = E[(k + n_embd/2)*n_codes + l];

            mag = exp(mag);

            if (mag > 1e2) {
                mag = 1e2;
            }
            S[2*(k*n_codes + l) + 0] = mag*cosf(phi);
            S[2*(k*n_codes + l) + 1] = mag*sinf(phi);
        }
    }

    for (int l = 0; l < n_codes; ++l) {
        for (int k = 0; k < n_embd/2; ++k) {
            ST[l*n_embd + 2*k + 0] = S[2*(k*n_codes + l) + 0];
            ST[l*n_embd + 2*k + 1] = S[2*(k*n_codes + l) + 1];
        }
    }

    std::vector<float> res  (n_codes*n_fft);
    std::vector<float> hann2(n_codes*n_fft);

    std::vector<std::thread> workers(n_thread);
    for (int i = 0; i < n_thread; ++i) {
        workers[i] = std::thread([&, i]() {
            for (int l = i; l < n_codes; l += n_thread) {
                irfft(n_fft, ST.data() + l*n_embd, res.data() + l*n_fft);
                for (int j = 0; j < n_fft; ++j) {
                    res  [l*n_fft + j] *= hann[j];
                    hann2[l*n_fft + j]  = hann[j] * hann[j];
                }
            }
        });
    }
    for (int i = 0; i < n_thread; ++i) {
        workers[i].join();
    }

    std::vector<float> audio;
    std::vector<float> env;

    fold(res,   n_out, n_win, n_hop, n_pad, audio);
    fold(hann2, n_out, n_win, n_hop, n_pad, env);

    for (size_t i = 0; i < audio.size(); ++i) {
        audio[i] /= env[i];
    }

    return audio;
}

// Forward declarations from rn-llama.h
extern bool rnllama_verbose;
void log(const char *level, const char *function, int line, const char *format, ...);

#define LOG_ERROR(MSG, ...) log("ERROR", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_WARNING(MSG, ...) log("WARNING", __func__, __LINE__, MSG, ##__VA_ARGS__)
#define LOG_INFO(MSG, ...) log("INFO", __func__, __LINE__, MSG, ##__VA_ARGS__)

// TTS member functions
tts_type llama_rn_context_tts::getTTSType(llama_rn_context* main_ctx, json speaker) {
    if (speaker.is_object() && speaker.contains("version")) {
        std::string version = speaker["version"].get<std::string>();
        if (version == "0.2") {
            return OUTETTS_V0_2;
        } else if (version == "0.3") {
            return OUTETTS_V0_3;
        } else {
            LOG_ERROR("Unsupported speaker version '%s'\n", version.c_str());
        }
    }
    if (type != UNKNOWN) {
        return type;
    }
    const char *chat_template = llama_model_chat_template(main_ctx->model, nullptr);
    if (chat_template && std::string(chat_template) == "outetts-0.3") {
        return OUTETTS_V0_3;
    }
    if (main_ctx->model->name.find("OuteTTS 0.1") != std::string::npos) {
        return OUTETTS_V0_1;
    }
    return OUTETTS_V0_2;
}

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(llama_rn_context* main_ctx, const std::string &speaker_json_str, const std::string &text_to_speak) {
    std::string audio_text = default_audio_text;
    std::string audio_data = default_audio_data;

    json speaker = speaker_json_str.empty() ? json::object() : json::parse(speaker_json_str);
    const tts_type tts_type = getTTSType(main_ctx, speaker);
    if (tts_type == UNKNOWN) {
        LOG_ERROR("Unknown TTS version");
        return {"", nullptr};
    }

    if (tts_type == OUTETTS_V0_3) {
        audio_text = std::regex_replace(audio_text, std::regex(R"(<\|text_sep\|>)"), "<|space|>");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_start\|>)"), "");
        audio_data = std::regex_replace(audio_data, std::regex(R"(<\|code_end\|>)"), "<|space|>");
    }

    if (!speaker_json_str.empty()) {
        audio_text = audio_text_from_speaker(speaker, tts_type);
        audio_data = audio_data_from_speaker(speaker, tts_type);
    }

    std::string prompt = "<|im_start|>\n" + audio_text + process_text(text_to_speak, tts_type) + "<|text_end|>\n" + audio_data + "\n";

    if (tts_type == OUTETTS_V0_1) {
        return {prompt, OUTETTS_V1_GRAMMAR};
    } else if (tts_type == OUTETTS_V0_2 || tts_type == OUTETTS_V0_3) {
        return {prompt, OUTETTS_V2_GRAMMAR};
    } else {
        return {prompt, nullptr};
    }
}

std::vector<llama_token> llama_rn_context_tts::getAudioCompletionGuideTokens(llama_rn_context* main_ctx, const std::string &text_to_speak) {
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    const tts_type tts_type = getTTSType(main_ctx);
    std::string clean_text = process_text(text_to_speak, tts_type);

    const std::string& delimiter = (tts_type == OUTETTS_V0_3 ? "<|space|>" : "<|text_sep|>");

    std::vector<llama_token> result;
    size_t start = 0;
    size_t end = clean_text.find(delimiter);

    //first token is always a newline, as it was not previously added
    result.push_back(common_tokenize(vocab, "\n", false, true)[0]);

    while (end != std::string::npos) {
        std::string current_word = clean_text.substr(start, end - start);
        auto tmp = common_tokenize(vocab, current_word, false, true);
        result.push_back(tmp[0]);
        start = end + delimiter.length();
        end = clean_text.find(delimiter, start);
    }

    // Add the last part
    std::string current_word = clean_text.substr(start);
    auto tmp = common_tokenize(vocab, current_word, false, true);
    if (tmp.size() > 0) {
        result.push_back(tmp[0]);
    }

    // Add Audio End, forcing stop generation
    result.push_back(common_tokenize(vocab, "<|audio_end|>", false, true)[0]);

    return result;
}

std::vector<float> llama_rn_context_tts::decodeAudioTokens(llama_rn_context* main_ctx, const std::vector<llama_token> &tokens) {
    std::vector<llama_token> tokens_audio = tokens;
    tts_type tts_type = getTTSType(main_ctx);
    if (tts_type == OUTETTS_V0_3 || tts_type == OUTETTS_V0_2) {
        tokens_audio.erase(std::remove_if(tokens_audio.begin(), tokens_audio.end(), [](llama_token t) { return t < 151672 || t > 155772; }), tokens_audio.end());
        for (auto & token : tokens_audio) {
            token -= 151672;
        }
    } else {
        LOG_ERROR("Unsupported audio tokens");
        return std::vector<float>();
    }
    const int n_codes = tokens_audio.size();
    llama_batch batch = llama_batch_init(n_codes, 0, 1);
    for (size_t i = 0; i < tokens_audio.size(); ++i) {
        llama_batch_add(&batch, tokens_audio[i], i, { 0 }, true);
    }
    if (batch.n_tokens != n_codes) {
        LOG_ERROR("batch.n_tokens != n_codes: %d != %d", batch.n_tokens, n_codes);
        return std::vector<float>();
    }
    if (llama_encode(ctx, batch) != 0) {
        LOG_ERROR("llama_encode() failed");
        return std::vector<float>();
    }
    llama_synchronize(ctx);
    const int n_embd = llama_model_n_embd(model);
    const float * embd = llama_get_embeddings(ctx);
    return embd_to_audio(embd, n_codes, n_embd, main_ctx->params.cpuparams.n_threads);
}

}
