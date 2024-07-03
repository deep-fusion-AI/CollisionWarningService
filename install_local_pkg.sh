#! /bin/bash
LOCAL_PYTHON3_SITE_PKG="/home/autoroad/.local/lib/python3.10/site-packages"

FCW_CORE=$PWD"/fcw-core/fcw_core"
FCW_SERVICE=$PWD"/fcw-service/fcw_service"
FCW_CLIENT=$PWD"/fcw-client/fcw_client"
FCW_UTILS=$PWD"/fcw-core-utils/fcw_core_utils"

#ln -s /home/dfai/dongho/CollisionWarningService/fcw-core/fcw_core ./

echo $FCW_CORE
echo $FCW_SERVICE
echo $FCW_CLIENT
echo $FCW_UTILS

# make link
ln -s $FCW_CORE $LOCAL_PYTHON3_SITE_PKG
ln -s $FCW_SERVICE $LOCAL_PYTHON3_SITE_PKG
ln -s $FCW_CLIENT $LOCAL_PYTHON3_SITE_PKG
ln -s $FCW_UTILS $LOCAL_PYTHON3_SITE_PKG


# check
ls "$LOCAL_PYTHON3_SITE_PKG" -al | grep fcw
