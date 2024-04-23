# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.4.0] - 2024-04-23
### Changed
- 更新依赖库版本.

## [0.3.9] - 2024-02-18
### Changed
- 更新依赖库版本.

## [0.3.8] - 2024-01-28
### Changed
- 更新依赖库版本.

## [0.3.7] - 2024-01-21
### Changed
- 如果是服务, 引用base1x.logger则用quant1x作为日志文件名.

## [0.3.6] - 2024-01-21
### Changed
- 修订用户宿主目录, 如果是~直接转换绝对路径.

## [0.3.5] - 2024-01-20
### Changed
- 修订export.

## [0.3.4] - 2024-01-20
### Changed
- 修订日志模块, 作为默认的日志记录器.
- !2 根据app名字初始化logger Merge pull request !2 from heathen666/master.
- 根据app名字初始化logger.

## [0.3.3] - 2024-01-10
### Changed
- 增加获取应用程序信息的功能.

## [0.3.2] - 2023-12-11
### Changed
- 拟增加yaml配置文件的自动装载功能.
- 调整依赖包.

## [0.3.1] - 2023-12-08
### Changed
- 去掉init中git相关的函数.

## [0.3.0] - 2023-12-01
### Changed
- 增加获取用户宿主目录的函数homedir.

## [0.2.9] - 2023-11-30
### Changed
- 增加二级市场函数-证券代码.

## [0.2.8] - 2023-11-30
### Changed
- 简化引入路径增加秒数转时间戳字符串的函数.

## [0.2.7] - 2023-11-30
### Changed
- 增加日志工具.

## [0.2.6] - 2023-11-30
### Changed
- 修改浮点四舍五入的函数名.

## [0.2.5] - 2023-11-30
### Changed
- 修改浮点四舍五入的函数名.

## [0.2.4] - 2023-11-30
### Changed
- 调整简化引入路径.
- 修改源文件名, datetime.py导致引入datetime失败.
- 新增.git路径检测机制.

## [0.2.3] - 2023-11-30
### Changed
- 修复路径可能存在非项目本地文件路径的问题.

## [0.2.2] - 2023-11-30
### Changed
- 调整简化引入的路径.
- 调整时间格式常量名.
- 调整datetime常量.
- 源文件名path.py不准确, 其实际作为开发时的python检索路径的前置处理.
- 新增: file类函数.
- 新增: num类函数.
- 新增: 设计模式.

## [0.2.1] - 2023-11-29
### Changed
- 简化path的函数引用路径.

## [0.2.0] - 2023-11-29
### Changed
- 调整get_lan_address函数名, 去掉前面的get_前缀.

## [0.1.9] - 2023-11-28
### Changed
- 增加cpu核数的相关函数.

## [0.1.8] - 2023-11-28
### Changed
- 增加获取本机局域网ip地址的函数.

## [0.1.7] - 2023-11-28
### Changed
- 新增IPv4地址检测功能, 总的原则是内网安全, 外网不安全.

## [0.1.6] - 2023-11-28
### Changed
- 调整打包配置, 删除rst格式的文档.
- 增加依赖库GitPython,版本3.1.5.
- 调整__init__中引用, 简化import路径.

## [0.1.5] - 2023-11-28
### Changed
- 增加单例模式, 时间范围, 交易时段.

## [0.1.4] - 2023-11-20
### Changed
- 获取项目路径增加默认路径.

## [0.1.3] - 2023-11-20
### Changed
- 调整包路径.
- 调整包路径.

## [0.1.2] - 2023-11-20
### Changed
- 调整包路径.

## [0.1.1] - 2023-11-20
### Changed
- 修订发布脚本, 增加清理发布过程中的临时文件夹.
- 修订发布脚本.

## [0.1.0] - 2023-11-20

### Changed

- 提交第一个版本.
- 修订README.
- Add LICENSE.
- Initial commit.

[Unreleased]: https://gitee.com/quant1x/base/compare/v0.4.0...HEAD

[0.4.0]: https://gitee.com/quant1x/base/compare/v0.3.9...v0.4.0
[0.3.9]: https://gitee.com/quant1x/base/compare/v0.3.8...v0.3.9
[0.3.8]: https://gitee.com/quant1x/base/compare/v0.3.7...v0.3.8
[0.3.7]: https://gitee.com/quant1x/base/compare/v0.3.6...v0.3.7
[0.3.6]: https://gitee.com/quant1x/base/compare/v0.3.5...v0.3.6
[0.3.5]: https://gitee.com/quant1x/base/compare/v0.3.4...v0.3.5
[0.3.4]: https://gitee.com/quant1x/base/compare/v0.3.3...v0.3.4
[0.3.3]: https://gitee.com/quant1x/base/compare/v0.3.2...v0.3.3
[0.3.2]: https://gitee.com/quant1x/base/compare/v0.3.1...v0.3.2
[0.3.1]: https://gitee.com/quant1x/base/compare/v0.3.0...v0.3.1
[0.3.0]: https://gitee.com/quant1x/base/compare/v0.2.9...v0.3.0
[0.2.9]: https://gitee.com/quant1x/base/compare/v0.2.8...v0.2.9
[0.2.8]: https://gitee.com/quant1x/base/compare/v0.2.7...v0.2.8
[0.2.7]: https://gitee.com/quant1x/base/compare/v0.2.6...v0.2.7
[0.2.6]: https://gitee.com/quant1x/base/compare/v0.2.5...v0.2.6
[0.2.5]: https://gitee.com/quant1x/base/compare/v0.2.4...v0.2.5
[0.2.4]: https://gitee.com/quant1x/base/compare/v0.2.3...v0.2.4
[0.2.3]: https://gitee.com/quant1x/base/compare/v0.2.2...v0.2.3
[0.2.2]: https://gitee.com/quant1x/base/compare/v0.2.1...v0.2.2
[0.2.1]: https://gitee.com/quant1x/base/compare/v0.2.0...v0.2.1
[0.2.0]: https://gitee.com/quant1x/base/compare/v0.1.9...v0.2.0
[0.1.9]: https://gitee.com/quant1x/base/compare/v0.1.8...v0.1.9
[0.1.8]: https://gitee.com/quant1x/base/compare/v0.1.7...v0.1.8
[0.1.7]: https://gitee.com/quant1x/base/compare/v0.1.6...v0.1.7
[0.1.6]: https://gitee.com/quant1x/base/compare/v0.1.5...v0.1.6
[0.1.5]: https://gitee.com/quant1x/base/compare/v0.1.4...v0.1.5
[0.1.4]: https://gitee.com/quant1x/base/compare/v0.1.3...v0.1.4
[0.1.3]: https://gitee.com/quant1x/base/compare/v0.1.2...v0.1.3
[0.1.2]: https://gitee.com/quant1x/base/compare/v0.1.1...v0.1.2
[0.1.1]: https://gitee.com/quant1x/base/compare/v0.1.0...v0.1.1
[0.1.0]: https://gitee.com/quant1x/base/releases/tag/v0.1.0
