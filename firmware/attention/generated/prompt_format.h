// flash.sh writes this for the model it flashes: how a request is wrapped before the model sees it, and what
// the board asks itself when left alone. This one is for the model in model/: "# <request>\n" -> a small program.
#define PROMPT_PREFIX "# "
#define PROMPT_SUFFIX "\n"
#define REQUEST_LIST "draw a smiley", "draw a star", "draw a house", "draw a sun", "draw a heart", "count to ten"
