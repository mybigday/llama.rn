import { StyleSheet } from 'react-native'
import { defaultTheme as defaultLightTheme, darkTheme as defaultDarkTheme } from  '@flyerhq/react-native-chat-ui'
import type { ThemeColors } from '../contexts/ThemeContext'

export const chatLightTheme = {
  ...defaultLightTheme,
  colors: {
    ...defaultLightTheme.colors,
    background: '#f0f0f0', // light grey
  },
}

export const chatDarkTheme = {
  ...defaultDarkTheme,
  colors: {
    ...defaultDarkTheme.colors,
    background: '#1f1c383a', // purple dark
  },
}

// Common spacing values
export const Spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
} as const

// Common font sizes
export const FontSizes = {
  small: 12,
  medium: 14,
  large: 16,
  xlarge: 18,
  xxlarge: 24,
} as const

// Function to create themed styles
export const createThemedStyles = (colors: ThemeColors) => {
  const isDark = colors.background === '#000000'

  return StyleSheet.create({
    // Container styles
    container: {
      flex: 1,
      backgroundColor: colors.background,
    },
    centerContainer: {
      flex: 1,
      alignItems: 'center',
      justifyContent: 'center',
    },
    setupContainer: {
      flex: 1,
      padding: Spacing.lg,
    },
    scrollContent: {
      paddingBottom: 20,
    },

    // Header styles
    header: {
      paddingHorizontal: Spacing.lg,
      paddingVertical: Spacing.md,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: isDark ? 4 : 3,
      elevation: 2,
    },
    headerRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    headerTitle: {
      fontSize: FontSizes.xlarge,
      fontWeight: '700',
      color: colors.text,
      textAlign: 'center',
    },
    headerButton: {
      marginRight: 4,
    },
    headerButtonText: {
      color: colors.primary,
      fontSize: FontSizes.large,
      fontWeight: '600',
    },

    // Text styles
    setupDescription: {
      fontSize: FontSizes.large,
      color: colors.textSecondary,
      lineHeight: 24,
      marginBottom: Spacing.xxl,
      textAlign: 'center',
    },
    description: {
      fontSize: FontSizes.medium,
      color: colors.textSecondary,
      lineHeight: 20,
      marginVertical: Spacing.lg,
      textAlign: 'center',
    },

    // Button styles
    button: {
      margin: 10,
      padding: 10,
      backgroundColor: colors.buttonBackground,
      borderRadius: 5,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.4 : 0.15,
      shadowRadius: isDark ? 4 : 3,
      elevation: 2,
    },
    buttonText: {
      color: colors.white,
      fontSize: FontSizes.large,
      fontWeight: '600',
    },
    primaryButton: {
      backgroundColor: colors.primary,
      paddingHorizontal: Spacing.xl,
      paddingVertical: Spacing.md,
      borderRadius: Spacing.sm,
      shadowColor: colors.primary,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.5 : 0.25,
      shadowRadius: isDark ? 6 : 4,
      elevation: 3,
    },
    primaryButtonText: {
      color: colors.white,
      fontSize: FontSizes.large,
      fontWeight: '700',
    },
    primaryButtonDisabled: {
      backgroundColor: colors.disabled,
    },
    primaryButtonActive: {
      backgroundColor: isDark ? '#0077DD' : '#0055AA',
      transform: [{ scale: 0.98 }],
      shadowOpacity: isDark ? 0.6 : 0.35,
    },
    secondaryButton: {
      paddingHorizontal: Spacing.md,
      paddingVertical: Spacing.xs,
      backgroundColor: colors.error,
      borderRadius: 6,
      shadowColor: colors.error,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.4 : 0.2,
      shadowRadius: isDark ? 4 : 3,
      elevation: 2,
    },
    secondaryButtonText: {
      color: colors.white,
      fontSize: FontSizes.medium,
      fontWeight: '600',
    },
    textButton: {
      fontSize: FontSizes.large,
      color: colors.primary,
      fontWeight: '500',
    },
    disabledButton: {
      opacity: 0.5,
    },

    // Loading styles
    loadingContainer: {
      alignItems: 'center',
      marginTop: Spacing.xxl,
    },
    loadingText: {
      marginTop: Spacing.sm,
      fontSize: FontSizes.large,
      color: colors.textSecondary,
    },

    // Progress bar styles
    progressContainer: {
      marginTop: Spacing.lg,
      width: '100%',
      alignItems: 'center',
    },
    progressBar: {
      width: '80%',
      height: 8,
      backgroundColor: colors.border,
      borderRadius: 4,
      overflow: 'hidden',
    },
    progressFill: {
      height: '100%',
      backgroundColor: colors.primary,
      borderRadius: 4,
    },

    // Card styles
    card: {
      backgroundColor: colors.card,
      borderRadius: Spacing.md,
      padding: Spacing.lg,
      marginVertical: Spacing.sm,
      marginHorizontal: Spacing.lg,
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.3 : 0.1,
      shadowRadius: isDark ? 6 : 4,
      elevation: 3,
    },

    // Form styles
    paramGroup: {
      backgroundColor: colors.surface,
      borderRadius: Spacing.md,
      padding: Spacing.lg,
      marginBottom: Spacing.md,
    },
    paramLabel: {
      fontSize: FontSizes.large,
      fontWeight: '600',
      color: colors.text,
      marginBottom: 4,
    },
    paramDescription: {
      fontSize: FontSizes.medium,
      color: colors.textSecondary,
      lineHeight: 18,
      marginBottom: Spacing.md,
    },
    textInput: {
      borderWidth: 1,
      borderColor: colors.border,
      borderRadius: Spacing.sm,
      paddingHorizontal: Spacing.md,
      paddingVertical: 10,
      fontSize: FontSizes.large,
      backgroundColor: colors.inputBackground,
      color: colors.text,
    },

    // Modal styles
    modalContainer: {
      flex: 1,
      backgroundColor: isDark ? 'rgba(0, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.5)',
      justifyContent: 'center',
      alignItems: 'center',
    },
    modalContent: {
      backgroundColor: colors.surface,
      borderRadius: Spacing.lg,
      padding: Spacing.xl,
      margin: Spacing.xl,
      maxHeight: '80%',
      maxWidth: '95%',
      minWidth: '85%',
      shadowColor: colors.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: isDark ? 0.6 : 0.25,
      shadowRadius: isDark ? 12 : 8,
      elevation: 8,
    },
    modalHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingHorizontal: Spacing.lg,
      paddingVertical: Spacing.md,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    modalTitle: {
      fontSize: FontSizes.xlarge,
      fontWeight: '700',
      color: colors.text,
    },
    modalCloseButton: {
      fontSize: FontSizes.xlarge,
      color: colors.primary,
      fontWeight: '600',
    },

    // Settings styles
    settingsContainer: {
      alignItems: 'center',
      marginVertical: Spacing.lg,
    },

    // Utility styles
    flexRow: {
      flexDirection: 'row',
    },
    flexColumn: {
      flexDirection: 'column',
    },
    alignCenter: {
      alignItems: 'center',
    },
    justifyCenter: {
      justifyContent: 'center',
    },
    justifyBetween: {
      justifyContent: 'space-between',
    },
    flex1: {
      flex: 1,
    },
    marginBottom: {
      marginBottom: Spacing.lg,
    },
    marginTop: {
      marginTop: Spacing.lg,
    },

    // Custom Model Styles
    modelSectionTitle: {
      fontSize: 20,
      fontWeight: '700',
      color: colors.text,
      marginTop: 8,
      marginBottom: 8,
      letterSpacing: -0.3,
    },
    addCustomModelButton: {
      backgroundColor: colors.primary,
      borderRadius: 12,
      paddingVertical: 14,
      paddingHorizontal: 20,
      marginHorizontal: 16,
      marginVertical: 12,
      shadowColor: colors.primary,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: isDark ? 0.4 : 0.3,
      shadowRadius: isDark ? 10 : 8,
      elevation: 4,
    },
    addCustomModelButtonText: {
      color: colors.white,
      fontSize: 16,
      fontWeight: '700',
      textAlign: 'center',
      letterSpacing: -0.2,
    },
    customModelDefaultSection: {
      marginTop: 8,
    },
  })
}
