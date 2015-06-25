`%>%` <- function(p, func) {
    p
    func <- substitute(func)
    class(func) <- 'list'
    env <- parent.frame()
    eval(as.call(c(func[1L], quote(.), func[-1L])),
         enclos = env, envir = list(. = p))
}
