@echo off
REM PyPi ģ�� �����ű�

REM ��ȡ��ǰ·��, ���ڷ���
echo "���ڴ��..."
python setup.py sdist bdist_wheel
echo "������"
echo "�����ϴ�PyPi.org..."
twine upload dist/*
echo "�ϴ����"
rmdir /S /Q dist
rmdir /S /Q build
rmdir /S /Q quant1x_base.egg-info
rmdir /S /Q .eggs
rem version=`python setup.py --version`
REM echo "�汾��: ${version}"
REM echo "git �����tag"
REM git tag -a v$version -m "Release version ${version}"
REM git push --tags
@echo on