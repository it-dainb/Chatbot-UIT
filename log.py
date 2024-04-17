import datetime


DEBUG = False

def log(args, level = 0):
    
    text = " ".join([str(i) for i in args])
    
    if text == "":
        # print()
        return
    
    currentDT = datetime.datetime.now()
    time = currentDT.strftime("%H:%M:%S")
    
    prefix = time + " "

    suffix = " "
    
    text = str(text)
    
    if level == -1:
        color = (255, 255, 255)
        mess = "DEBUG :"
    elif level == 0:
        color = (0, 255, 0)
        mess = "INFO  :"
    elif level == 1:
        color = (255, 255, 0)
        mess = "WARN  :"
    else:
        color = (255, 0, 0)
        mess = "ERROR :"
    
    r, g, b = color
    
    text = str(prefix) + mess + str(suffix) + text
    
    if level == -1:
        if DEBUG:
            print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")
    else:
        print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")

def debug(*args):
    log(args, level = -1)

def info(*args):
    log(args, level = 0)

def warn(*args):
    log(args, level = 1)
    
def error(*args):
    log(args, level = 2)

if __name__ == "__main__":
    debug("")
    debug("hello", 10, [1,2,3], {1:2, 3:4})
    info("hello")
    warn("hello")
    error("hello")