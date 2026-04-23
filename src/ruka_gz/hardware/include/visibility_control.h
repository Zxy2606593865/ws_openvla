#ifndef RUKA__VISIBILITY_CONTROL_H_
#define RUKA__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define RUKA_EXPORT __attribute__((dllexport))
#define RUKA_IMPORT __attribute__((dllimport))
#else
#define RUKA_EXPORT __declspec(dllexport)
#define RUKA_IMPORT __declspec(dllimport)
#endif
#ifdef RUKA_BUILDING_DLL
#define RUKA_PUBLIC RUKA_EXPORT
#else
#define RUKA_PUBLIC RUKA_IMPORT
#endif
#define RUKA_PUBLIC_TYPE RUKA_PUBLIC
#define RUKA_LOCAL
#else
#define RUKA_EXPORT __attribute__((visibility("default")))
#define RUKA_IMPORT
#if __GNUC__ >= 4
#define RUKA_PUBLIC __attribute__((visibility("default")))
#define RUKA_LOCAL __attribute__((visibility("hidden")))
#else
#define RUKA_PUBLIC
#define RUKA_LOCAL
#endif
#define RUKA_PUBLIC_TYPE
#endif

#endif  
