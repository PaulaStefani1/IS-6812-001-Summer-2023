library(polite)
bow("https://amazon.com/s?k=best+seller+book+list&page=")

library(ralger)
my_link <- "https://amazon.com/s?k=best+seller+book+list&page=1"
my_node<-".a-color-base.a-text-normal"# selector gadget will give these element ID
best_books <- scrap(link = my_link, node = my_node)
head(best_books,5)

base_link <- "https://amazon.com/s?k=best+seller+book+list&page="
links <- paste0(base_link, 1:7) # there are 7 pages
node<-".a-color-base.a-text-normal"

all_books<-scrap(links, node)
head(all_books)

base_link <- "https://www.amazon.com/s?k=best+seller+book+list+2021&page="
links <- paste0(base_link, 1:4) # tlet's take 4 pages

my_nodes <- c(
  ".a-color-base.a-text-normal", # The title
  ".aok-align-bottom", # Rating
  ".a-size-small .a-link-normal .a-size-base", #Number of ratings
  ".a-price-whole") # Price whole
  
names <- c("title", "rating", "number of ratings", "price ") # respect the nodes order

fullds<-tidy_scrap(link = links, nodes = my_nodes, colnames = names)
head(fullds,5)

bow("https://www.iweblists.com/us/commerce/MarketCapitalization.html")

data <- table_scrap(link ="https://www.iweblists.com/us/commerce/MarketCapitalization.html")
head(data)
