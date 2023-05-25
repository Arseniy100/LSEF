install.packages("torch")



install.packages("shiny")
install.packages("learnr")
install.packages("LibLantern")
# install.packages("torch")
#Sys.getenv("http_proxy")


# options(internet.info = 0)

# install.packages("C:\\Users\\user410\\Python\\lsm_s2\\R from MD\\packages_for_torch\\bit_4.0.4.tar.gz", repos = NULL, type="source")
# install.packages("C:\\Users\\user410\\Python\\lsm_s2\\R from MD\\packages_for_torch_1\\libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip", repos = NULL, type="source")
# install.packages("luz")
library(torch)
install_torch()

torch_tensor(1)

install_torch_from_file(
  libtorch = "file://C:/Users/user410/Python/lsm_s2/R from MD/packages_for_torch/libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip",
  liblantern = "file://C:/Users/user410/Python/lsm_s2/R from MD/packages_for_torch/Linux-gpu-102.zip"
  )

remove.packages("torch")

getwd()

get_install_libs_url(type = "1.11.0")
sessionInfo()
