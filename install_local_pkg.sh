LOCAL_PYTHON3_SITE_PKG="~/.local/lib/python3.10/site-packages"

FCW_CORE=$PWD"/fcw-core/fcw_core"
FCW_SERVICE=$PWD"/fcw-service/fcw_service"
FCW_CLIENT=$PWD"/fcw-client/fcw_client"

#ln -s /home/dfai/dongho/CollisionWarningService/fcw-core/fcw_core ./

echo $FCW_CORE
echo $FCW_SERVICE
echo $FCW_CLIENT

# make link
cd $LOCAL_PYTHON3_SITE_PKG
ln -s $FCW_CORE ./
ln -s $FCW_SERVICE ./
ln -s $FCW_CLIENT ./

# check
ls -al | grep fcw