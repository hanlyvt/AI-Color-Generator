"use client";
import { useState } from "react";

export default function ColorMatcher() {
  const [text, setText] = useState("");
  const [description, setDescription] = useState("");
  const [colors, setColors] = useState([]);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [showKeywords, setShowKeywords] = useState(false);
  const [inputMode, setInputMode] = useState("keywords"); // 'keywords' or 'description'

  const matchColors = async () => {
    const searchText = inputMode === "keywords" ? text : description;
    if (!searchText.trim()) return;

    setLoading(true);
    try {
      const response = await fetch("/api/colors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: searchText, top_k: 5 }),
      });
      const data = await response.json();
      setColors(data.matches);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (hex, index) => {
    navigator.clipboard.writeText(hex);
    setCopied(index);
    setTimeout(() => setCopied(null), 2000);
  };

  const suggestions = [
    "innovative tech startup",
    "luxury elegant brand",
    "eco-friendly sustainable",
    "playful creative",
    "calm peaceful spa",
    "energetic sports",
  ];

  const keywordCategories = [
    {
      name: "Emoties",
      keywords: [
        "energetic",
        "calm",
        "playful",
        "serious",
        "elegant",
        "friendly",
        "professional",
        "warm",
        "cool",
        "bold",
        "subtle",
        "confident",
        "peaceful",
      ],
    },
    {
      name: "Stijl",
      keywords: [
        "modern",
        "vintage",
        "minimalist",
        "luxurious",
        "rustic",
        "futuristic",
        "classic",
        "artistic",
        "corporate",
        "casual",
        "sophisticated",
      ],
    },
    {
      name: "Sector",
      keywords: [
        "tech",
        "healthcare",
        "finance",
        "education",
        "retail",
        "hospitality",
        "creative",
        "sports",
        "food",
        "fashion",
        "nature",
        "wellness",
      ],
    },
    {
      name: "Waarden",
      keywords: [
        "sustainable",
        "innovative",
        "trustworthy",
        "dynamic",
        "reliable",
        "authentic",
        "premium",
        "accessible",
        "inclusive",
        "traditional",
      ],
    },
  ];

  return (
    <div
      className={`h-screen overflow-hidden transition-colors duration-500 ${
        darkMode
          ? "bg-gradient-to-br from-gray-900 via-red-950 to-gray-900"
          : "bg-gradient-to-br from-red-50 via-orange-50 to-pink-50"
      }`}
    >
      {/* Animated background blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className={`absolute -top-40 -right-40 w-96 h-96 ${
            darkMode ? "bg-red-900/30" : "bg-red-300/50"
          } rounded-full mix-blend-multiply filter blur-3xl animate-blob`}
        ></div>
        <div
          className={`absolute -bottom-40 -left-40 w-96 h-96 ${
            darkMode ? "bg-orange-900/30" : "bg-orange-300/50"
          } rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000`}
        ></div>
        <div
          className={`absolute top-1/2 left-1/2 w-96 h-96 ${
            darkMode ? "bg-pink-900/30" : "bg-pink-300/50"
          } rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000`}
        ></div>
      </div>

      {/* Dark mode toggle */}
      <button
        onClick={() => setDarkMode(!darkMode)}
        className={`fixed top-6 right-6 z-50 ${
          darkMode ? "bg-gray-800/80" : "bg-white/80"
        } backdrop-blur-xl rounded-full p-3 shadow-2xl hover:scale-110 transition-all duration-300 border ${
          darkMode ? "border-red-700/50" : "border-red-200"
        }`}
      >
        {darkMode ? (
          <span className="text-2xl">☀️</span>
        ) : (
          <span className="text-2xl">🌙</span>
        )}
      </button>

      <div className="relative h-full flex items-center justify-center px-6 py-6">
        {colors.length === 0 ? (
          /* Input view */
          <div className="w-full max-w-6xl grid grid-cols-3 gap-6 h-[85vh]">
            {/* Left: Keywords List */}
            <div
              className={`${
                darkMode
                  ? "bg-gray-900/60 border-red-800/30"
                  : "bg-white/70 border-red-200"
              } backdrop-blur-2xl rounded-3xl shadow-2xl p-6 border-2 overflow-y-auto`}
            >
              <div className="flex items-center justify-between mb-4">
                <h3
                  className={`text-xl font-black ${
                    darkMode ? "text-white" : "text-gray-900"
                  }`}
                >
                  📚 Keywords Bibliotheek
                </h3>
              </div>

              <p
                className={`text-sm ${
                  darkMode ? "text-gray-400" : "text-gray-600"
                } mb-4`}
              >
                Klik op keywords om ze toe te voegen aan je zoekopdracht
              </p>

              <div className="space-y-4">
                {keywordCategories.map((category, idx) => (
                  <div key={idx}>
                    <h4
                      className={`text-sm font-bold ${
                        darkMode ? "text-red-400" : "text-red-600"
                      } mb-2 flex items-center gap-2`}
                    >
                      <span className="w-2 h-2 rounded-full bg-gradient-to-r from-red-500 to-pink-500"></span>
                      {category.name}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {category.keywords.map((keyword, i) => (
                        <button
                          key={i}
                          onClick={() => {
                            if (inputMode === "keywords") {
                              setText(text ? `${text} ${keyword}` : keyword);
                            }
                          }}
                          className={`text-xs px-3 py-1.5 ${
                            darkMode
                              ? "bg-gray-800/60 hover:bg-red-900/50 text-gray-300 border-gray-700"
                              : "bg-gray-50 hover:bg-red-50 text-gray-700 border-gray-200"
                          } border rounded-full transition-all hover:scale-105 font-medium`}
                        >
                          {keyword}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Center: Main Input */}
            <div className="col-span-2 flex flex-col">
              <div className="text-center mb-8">
                <div className="inline-block mb-3 animate-bounce">
                  <span className="text-6xl drop-shadow-2xl">🎨</span>
                </div>
                <h1
                  className={`text-5xl font-black mb-3 bg-gradient-to-r ${
                    darkMode
                      ? "from-red-400 via-orange-400 to-pink-400"
                      : "from-red-600 via-orange-600 to-pink-600"
                  } bg-clip-text text-transparent animate-gradient drop-shadow-xl`}
                >
                  AI Color Generator
                </h1>
                <p
                  className={`text-lg ${
                    darkMode ? "text-gray-300" : "text-gray-700"
                  }`}
                >
                  Ontdek de perfecte kleuren voor jouw project
                </p>
              </div>

              <div
                className={`${
                  darkMode
                    ? "bg-gray-900/60 border-red-800/30"
                    : "bg-white/70 border-red-200"
                } backdrop-blur-2xl rounded-3xl shadow-2xl p-8 border-2 flex-1 flex flex-col`}
              >
                {/* Toggle between input modes */}
                <div className="flex gap-2 mb-6">
                  <button
                    onClick={() => setInputMode("keywords")}
                    className={`flex-1 py-3 px-4 rounded-xl font-bold transition-all ${
                      inputMode === "keywords"
                        ? `${
                            darkMode
                              ? "bg-gradient-to-r from-red-600 to-pink-600 text-white"
                              : "bg-gradient-to-r from-red-500 to-pink-500 text-white"
                          } shadow-lg`
                        : `${
                            darkMode
                              ? "bg-gray-800/50 text-gray-400"
                              : "bg-gray-100 text-gray-600"
                          }`
                    }`}
                  >
                    🏷️ Keywords
                  </button>
                  <button
                    onClick={() => setInputMode("description")}
                    className={`flex-1 py-3 px-4 rounded-xl font-bold transition-all ${
                      inputMode === "description"
                        ? `${
                            darkMode
                              ? "bg-gradient-to-r from-red-600 to-pink-600 text-white"
                              : "bg-gradient-to-r from-red-500 to-pink-500 text-white"
                          } shadow-lg`
                        : `${
                            darkMode
                              ? "bg-gray-800/50 text-gray-400"
                              : "bg-gray-100 text-gray-600"
                          }`
                    }`}
                  >
                    💼 Bedrijfsbeschrijving
                  </button>
                </div>

                {inputMode === "keywords" ? (
                  /* Keywords input */
                  <div className="flex-1 flex flex-col">
                    <label
                      className={`block text-sm font-black ${
                        darkMode ? "text-red-400" : "text-red-600"
                      } mb-3 uppercase tracking-wider flex items-center gap-2`}
                    >
                      <span className="text-xl">✨</span> Keywords
                    </label>
                    <div className="relative flex-1 flex flex-col">
                      <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="bijv. innovative tech modern elegant..."
                        className={`flex-1 ${
                          darkMode
                            ? "bg-gray-800/80 border-red-700/50 text-white placeholder-gray-500 focus:border-red-500"
                            : "bg-white border-red-300 text-gray-900 placeholder-gray-400 focus:border-red-600"
                        } border-2 rounded-2xl p-4 text-lg focus:ring-4 ${
                          darkMode
                            ? "focus:ring-red-900/50"
                            : "focus:ring-red-100"
                        } focus:outline-none transition-all shadow-inner resize-none`}
                        rows="3"
                      />
                      {text && (
                        <button
                          onClick={() => setText("")}
                          className={`absolute right-4 top-4 ${
                            darkMode
                              ? "text-gray-500 hover:text-gray-300"
                              : "text-gray-400 hover:text-gray-600"
                          } transition`}
                        >
                          ✕
                        </button>
                      )}
                    </div>

                    <div className="mt-4 flex flex-wrap gap-2">
                      <span
                        className={`text-xs ${
                          darkMode ? "text-gray-400" : "text-gray-600"
                        } font-semibold`}
                      >
                        Snel proberen:
                      </span>
                      {suggestions.map((suggestion, i) => (
                        <button
                          key={i}
                          onClick={() => setText(suggestion)}
                          className={`text-sm px-3 py-1.5 ${
                            darkMode
                              ? "bg-gradient-to-r from-red-900/50 to-pink-900/50 hover:from-red-800/70 hover:to-pink-800/70 text-red-200 border border-red-700/30"
                              : "bg-gradient-to-r from-red-100 to-pink-100 hover:from-red-200 hover:to-pink-200 text-red-700 border border-red-200"
                          } rounded-full transition-all hover:scale-105 font-medium shadow-md`}
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  /* Description input */
                  <div className="flex-1 flex flex-col">
                    <label
                      className={`block text-sm font-black ${
                        darkMode ? "text-red-400" : "text-red-600"
                      } mb-3 uppercase tracking-wider flex items-center gap-2`}
                    >
                      <span className="text-xl">💼</span> Vertel over je bedrijf
                    </label>
                    <div className="relative flex-1 flex flex-col">
                      <textarea
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="Beschrijf je bedrijf, doelgroep, waarden en sfeer... bijv: 'Wij zijn een duurzame koffiebar voor studenten die een gezellige, warme en moderne sfeer willen creëren...'"
                        className={`flex-1 ${
                          darkMode
                            ? "bg-gray-800/80 border-red-700/50 text-white placeholder-gray-500 focus:border-red-500"
                            : "bg-white border-red-300 text-gray-900 placeholder-gray-400 focus:border-red-600"
                        } border-2 rounded-2xl p-4 text-lg focus:ring-4 ${
                          darkMode
                            ? "focus:ring-red-900/50"
                            : "focus:ring-red-100"
                        } focus:outline-none transition-all shadow-inner resize-none`}
                        rows="6"
                      />
                      {description && (
                        <button
                          onClick={() => setDescription("")}
                          className={`absolute right-4 top-4 ${
                            darkMode
                              ? "text-gray-500 hover:text-gray-300"
                              : "text-gray-400 hover:text-gray-600"
                          } transition`}
                        >
                          ✕
                        </button>
                      )}
                    </div>
                    <p
                      className={`text-xs ${
                        darkMode ? "text-gray-500" : "text-gray-500"
                      } mt-2`}
                    >
                      💡 Tip: Beschrijf je bedrijf zo gedetailleerd mogelijk
                      voor de beste resultaten
                    </p>
                  </div>
                )}

                <button
                  onClick={matchColors}
                  disabled={
                    loading ||
                    (inputMode === "keywords"
                      ? !text.trim()
                      : !description.trim())
                  }
                  className={`w-full mt-6 ${
                    darkMode
                      ? "bg-gradient-to-r from-red-600 via-orange-600 to-pink-600 hover:from-red-700 hover:via-orange-700 hover:to-pink-700"
                      : "bg-gradient-to-r from-red-500 via-orange-500 to-pink-500 hover:from-red-600 hover:via-orange-600 hover:to-pink-600"
                  } disabled:from-gray-500 disabled:to-gray-600 text-white font-black text-lg py-5 px-8 rounded-2xl transition-all shadow-2xl hover:shadow-red-500/50 hover:scale-[1.02] active:scale-95 disabled:cursor-not-allowed`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <svg
                        className="animate-spin h-6 w-6 mr-3"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                          fill="none"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        />
                      </svg>
                      AI aan het werk...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <span className="text-2xl">🚀</span>
                      Genereer Kleurenpalet
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Results view - Same as before */
          <div className="w-full h-full flex flex-col max-w-[95%] mx-auto animate-fadeIn">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2
                  className={`text-3xl font-black ${
                    darkMode ? "text-white" : "text-gray-900"
                  }`}
                >
                  Jouw Perfecte Palette
                </h2>
                <p
                  className={`${
                    darkMode ? "text-gray-400" : "text-gray-600"
                  } mt-1`}
                >
                  Gegenereerd voor:{" "}
                  <span
                    className={`font-bold ${
                      darkMode ? "text-red-400" : "text-red-600"
                    }`}
                  >
                    "
                    {inputMode === "keywords"
                      ? text
                      : description.substring(0, 50) + "..."}
                    "
                  </span>
                </p>
              </div>
              <button
                onClick={() => setColors([])}
                className={`${
                  darkMode
                    ? "bg-gray-800/80 border-red-700/30 hover:bg-gray-700/90 text-white"
                    : "bg-white/80 border-red-200 hover:bg-white text-gray-900"
                } backdrop-blur-xl rounded-xl shadow-xl px-6 py-3 border-2 font-bold transition-all hover:scale-105`}
              >
                ← Nieuw
              </button>
            </div>

            <div
              className={`${
                darkMode
                  ? "bg-gray-900/60 border-red-800/30"
                  : "bg-white/70 border-red-200"
              } backdrop-blur-2xl rounded-2xl shadow-xl p-4 mb-6 border-2`}
            >
              <div className="flex rounded-xl overflow-hidden h-20 shadow-lg">
                {colors.map((color, i) => (
                  <div
                    key={i}
                    style={{ backgroundColor: color.color_hex }}
                    className="flex-1 hover:flex-[1.5] transition-all duration-300 cursor-pointer relative group"
                    onClick={() => copyToClipboard(color.color_hex, i)}
                  >
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition flex items-center justify-center">
                      <span className="text-white font-mono font-bold text-sm opacity-0 group-hover:opacity-100 transition bg-black/70 px-3 py-2 rounded-lg backdrop-blur-sm shadow-xl">
                        {copied === i ? "✓ Copied!" : color.color_hex}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-5 gap-4 flex-1 overflow-hidden pb-4">
              {colors.map((color, i) => (
                <div
                  key={i}
                  className={`${
                    darkMode
                      ? "bg-gray-900/60 border-red-800/30"
                      : "bg-white/70 border-red-200"
                  } backdrop-blur-2xl rounded-2xl shadow-xl overflow-hidden hover:shadow-2xl transition-all hover:scale-[1.02] border-2 flex flex-col`}
                >
                  <div
                    style={{ backgroundColor: color.color_hex }}
                    className="h-32 relative group cursor-pointer"
                    onClick={() => copyToClipboard(color.color_hex, i)}
                  >
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition"></div>
                    <div
                      className={`absolute top-3 left-3 ${
                        darkMode ? "bg-gray-900/95" : "bg-white/95"
                      } backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg`}
                    >
                      <span
                        className={`text-base font-black bg-gradient-to-r ${
                          darkMode
                            ? "from-red-400 to-pink-400"
                            : "from-red-600 to-pink-600"
                        } bg-clip-text text-transparent`}
                      >
                        #{i + 1}
                      </span>
                    </div>
                  </div>

                  <div className="p-4 flex-1 flex flex-col">
                    <button
                      onClick={() => copyToClipboard(color.color_hex, i)}
                      className={`${
                        darkMode
                          ? "bg-gray-800/80 hover:bg-gray-700 border-red-700/30"
                          : "bg-gray-50 hover:bg-gray-100 border-red-200"
                      } border-2 rounded-lg px-3 py-2 mb-3 text-center font-mono font-bold text-sm ${
                        darkMode ? "text-white" : "text-gray-900"
                      } transition shadow-inner`}
                    >
                      {copied === i ? "✓ Copied!" : color.color_hex}
                    </button>

                    <h3
                      className={`text-base font-bold ${
                        darkMode ? "text-white" : "text-gray-900"
                      } mb-3 line-clamp-2 leading-tight`}
                    >
                      {color.color_name}
                    </h3>

                    <div className="mb-3">
                      <div className="flex justify-between items-center mb-1.5">
                        <span
                          className={`text-xs font-bold ${
                            darkMode ? "text-gray-400" : "text-gray-600"
                          }`}
                        >
                          AI Match
                        </span>
                        <span
                          className={`text-base font-black bg-gradient-to-r ${
                            darkMode
                              ? "from-red-400 to-pink-400"
                              : "from-red-600 to-pink-600"
                          } bg-clip-text text-transparent`}
                        >
                          {(color.score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div
                        className={`w-full ${
                          darkMode ? "bg-gray-800" : "bg-gray-200"
                        } rounded-full h-2 overflow-hidden shadow-inner`}
                      >
                        <div
                          className="bg-gradient-to-r from-red-500 via-orange-500 to-pink-500 h-2 rounded-full transition-all duration-1000 shadow-lg"
                          style={{ width: `${color.score * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-2 text-xs flex-1">
                      {color.emotion && (
                        <div className="flex items-start gap-1.5">
                          <span className="text-base">💭</span>
                          <div className="flex-1">
                            <div
                              className={`text-xs font-bold ${
                                darkMode ? "text-gray-500" : "text-gray-500"
                              } mb-0.5`}
                            >
                              EMOTIE
                            </div>
                            <div
                              className={`text-xs ${
                                darkMode ? "text-gray-300" : "text-gray-700"
                              } font-medium line-clamp-1`}
                            >
                              {color.emotion}
                            </div>
                          </div>
                        </div>
                      )}
                      {color.use_case && (
                        <div className="flex items-start gap-1.5">
                          <span className="text-base">💡</span>
                          <div className="flex-1">
                            <div
                              className={`text-xs font-bold ${
                                darkMode ? "text-gray-500" : "text-gray-500"
                              } mb-0.5`}
                            >
                              GEBRUIK
                            </div>
                            <div
                              className={`text-xs ${
                                darkMode ? "text-gray-300" : "text-gray-700"
                              } font-medium line-clamp-2`}
                            >
                              {color.use_case}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style jsx global>{`
        @keyframes gradient {
          0%,
          100% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
        }
        .animate-gradient {
          background-size: 200% 200%;
          animation: gradient 3s ease infinite;
        }
        @keyframes blob {
          0%,
          100% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(20px, -50px) scale(1.1);
          }
          50% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          75% {
            transform: translate(50px, 50px) scale(1.05);
          }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}
